import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque

import torch
import torchvision
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

from stp3.utils.geometry import (
    update_intrinsics,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
)
import stp3.utils.sampler as trajectory_sampler

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner
from stp3.trainer import TrainingModule
from stp3.datas.CarlaData import scale_and_crop_image

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
    return 'MVPAgent'

def make_contour(img, colour=[0, 0, 0], double_line=False):
    h, w = img.shape[:2]
    out = img.copy()
    # Vertical lines
    out[np.arange(h), np.repeat(0, h)] = colour
    out[np.arange(h), np.repeat(w - 1, h)] = colour

    # Horizontal lines
    out[np.repeat(0, w), np.arange(w)] = colour
    out[np.repeat(h - 1, w), np.arange(w)] = colour

    if double_line:
        out[np.arange(h), np.repeat(1, h)] = colour
        out[np.arange(h), np.repeat(w - 2, h)] = colour

        # Horizontal lines
        out[np.repeat(1, w), np.arange(w)] = colour
        out[np.repeat(h - 2, w), np.arange(w)] = colour
    return out

class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class MVPAgent(autonomous_agent.AutonomousAgent):
    def setup(self, checkpoint_path):
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.input_buffer = {'rgb': deque(maxlen=3), 'rgb_left': deque(maxlen=3), 'rgb_right': deque(maxlen=3),
                             'rgb_rear': deque(maxlen=3), 'gps': deque(maxlen=3), 'thetas': deque(maxlen=3)}

        trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=False)
        trainer.eval()
        trainer.cuda()
        self.model = trainer.model
        self.cfg = self.model.cfg

        self.save_path = None
        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            print(string)

            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

            (self.save_path / 'meta').mkdir()
            (self.save_path / 'show').mkdir()

        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )
        self.last_steer = 0

        self.turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3,n=40)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0,n=40)


        self.sequence_length = self.cfg.TIME_RECEPTIVE_FIELD + self.cfg.N_FUTURE_FRAMES
        self.receptive_field = self.cfg.TIME_RECEPTIVE_FIELD
        self.n_samples = self.cfg.PLANNING.SAMPLE_NUM

    def _init(self):
        self._route_planner = RoutePlanner(1.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale

        return gps

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_left'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_right'
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -1.3, 'y': 0.0, 'z': 2.3,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 400, 'height': 300, 'fov': 100,
                'id': 'rgb_rear'
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu'
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps'
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed'
            }
        ]

    def get_cam_para(self):
        def get_cam_to_ego(dof):
            yaw = dof[5]
            rotation = Quaternion(scalar=np.cos(yaw/2), vector=[0, 0, np.sin(yaw/2)])
            translation = np.array(dof[:3])[:, None]
            cam_to_ego = np.vstack([
                np.hstack((rotation.rotation_matrix,translation)),
                np.array([0,0,0,1])
            ])
            return cam_to_ego

        cam_front = [1.3, 0.0, 2.3, 0.0, 0.0, 0.0] # x,y,z,roll,pitch, yaw
        cam_left = [1.3, 0.0, 2.3, 0.0, 0.0, -60.0]
        cam_right = [1.3, 0.0, 2.3, 0.0, 0.0, 60.0]
        cam_rear = [-1.3, 0.0, 2.3, 0.0, 0.0, 180.0]
        front_to_ego = torch.from_numpy(get_cam_to_ego(cam_front)).float().unsqueeze(0)
        left_to_ego = torch.from_numpy(get_cam_to_ego(cam_left)).float().unsqueeze(0)
        right_to_ego = torch.from_numpy(get_cam_to_ego(cam_right)).float().unsqueeze(0)
        rear_to_ego = torch.from_numpy(get_cam_to_ego(cam_rear)).float().unsqueeze(0)
        extrinsic = torch.cat([front_to_ego, left_to_ego, right_to_ego, rear_to_ego], dim=0)

        sensor_data = {
            'width': 400,
            'height': 300,
            'fov': 100
        }
        w = sensor_data['width']
        h = sensor_data['height']
        fov = sensor_data['fov']
        f = w / (2 * np.tan(fov * np.pi/ 360))
        Cu = w / 2
        Cv = h / 2
        intrinsic = torch.Tensor([
            [f, 0, Cu],
            [0, f, Cv],
            [0, 0, 1]
        ])
        intrinsic = update_intrinsics(
            intrinsic, (h-256)/2, (w-256)/2,
            scale_width=1,
            scale_height=1
        )
        intrinsic = intrinsic.unsqueeze(0).expand(4,3,3)

        return extrinsic, intrinsic

    def get_future_egomotion(self, seq_x, seq_y, seq_theta):
        future_egomotions = []

        def convert_to_matrix_numpy(x, y, theta):
            matrix = np.zeros((4,4), dtype=np.float32)
            matrix[:2, :2] = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            matrix[2,2] = 1
            matrix[0,3] = x
            matrix[1,3] = y
            matrix[3,3] = 1
            return matrix

        for i in range(len(seq_x)-1):
            egopose_t0 = convert_to_matrix_numpy(seq_x[i], seq_y[i], seq_theta[i])
            egopose_t1 = convert_to_matrix_numpy(seq_x[i+1], seq_y[i+1], seq_theta[i+1])

            future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
            future_egomotion[3, :3] = 0.0
            future_egomotion[3, 3] = 1.0

            future_egomotion = torch.Tensor(future_egomotion).float()
            future_egomotion = mat2pose_vec(future_egomotion)
            future_egomotions.append(future_egomotion.unsqueeze(0))

        return torch.cat(future_egomotions, dim=0)

    def get_trajectory_sampling(self, v0, steering):

        Kappa = 2 * steering / 2.588

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        t_start = 0  # second
        t_end = self.cfg.N_FUTURE_FRAMES * 0.5  # second
        t_interval = 0.5 / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(v0, Kappa, T0, N0, tt, self.n_samples)
        sampled_trajectories = sampled_trajectories_fine[:, ::10]
        return sampled_trajectories

    def control_pid(self, waypoints, velocity, tick_data=None):
        '''
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert (waypoints.size(0) == 1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        # waypoints[:, 0] *= -1
        speed = velocity[0].data.cpu().numpy()

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = (speed / desired_speed) > 1.2

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(waypoints[3].astype(np.float64)),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'command': tick_data['next_command'],
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]

        result = {
            'rgb': rgb,
            'rgb_left': rgb_left,
            'rgb_right': rgb_right,
            'rgb_rear': rgb_rear,
            'gps': gps,
            'speed': speed,
            'compass': compass,
        }

        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value

        theta = compass + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        local_command_point = local_command_point * [1.0, -1.0]
        result['target_point'] = torch.from_numpy(local_command_point)

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        if self.step < 4:
            rgb = self.normalise_image(np.array(
                scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))

            rgb_left = self.normalise_image(np.array(
                scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))

            rgb_right = self.normalise_image(np.array(
                scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

            rgb_rear = self.normalise_image(np.array(
                scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

            self.input_buffer['thetas'].append(tick_data['compass'])
            self.input_buffer['gps'].append(tick_data['gps'])

            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0

            return control

        gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
        command = []
        if tick_data['next_command'] == 1:
            command.append('LEFT')
        elif tick_data['next_command'] == 2:
            command.append('RIGHT')
        elif tick_data['next_command'] == 3:
            command.append('FORWARD')
        else:
            command.append('LANE')
        target_points = tick_data['target_point'].to('cuda', dtype=torch.float32).unsqueeze(0)

        images = []
        rgb = self.normalise_image(np.array(
            scale_and_crop_image(Image.fromarray(tick_data['rgb']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
        images.append(torch.cat([p for p in self.input_buffer['rgb']], dim=1))

        rgb_left = self.normalise_image(np.array(
            scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
        images.append(torch.cat([p for p in self.input_buffer['rgb_left']], dim=1))

        rgb_right = self.normalise_image(np.array(
            scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))
        images.append(torch.cat([p for p in self.input_buffer['rgb_right']], dim=1))

        rgb_rear = self.normalise_image(np.array(
            scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), scale=1, crop=256))).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))
        images.append(torch.cat([p for p in self.input_buffer['rgb_rear']], dim=1))

        self.input_buffer['thetas'].append(tick_data['compass'])
        self.input_buffer['gps'].append(tick_data['gps'])

        images = torch.cat(images, dim=2).to('cuda', dtype=torch.float32) # (1,3,4,256,256)
        extrinsics, intrinsics = self.get_cam_para()
        extrinsics = extrinsics.unsqueeze(0).expand(3,4,4,4).unsqueeze(0).to('cuda', dtype=torch.float32)
        intrinsics = intrinsics.unsqueeze(0).expand(3,4,3,3).unsqueeze(0).to('cuda', dtype=torch.float32)
        future_egomotion = self.get_future_egomotion(
            seq_x=[p[0] for p in self.input_buffer['gps']],
            seq_y=[p[1] for p in self.input_buffer['gps']],
            seq_theta=[p for p in self.input_buffer['thetas']]
        ).to('cuda', dtype=torch.float32).unsqueeze(0)
        # print(self.input_buffer['gps'])
        trajs = torch.from_numpy(self.get_trajectory_sampling(tick_data['speed'], self.last_steer)).to('cuda', dtype=torch.float32).unsqueeze(0)


        output = self.model(
            images, intrinsics, extrinsics, future_egomotion,
        )
        n_present = self.model.receptive_field
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        pedestrian_prediction = output['pedestrian'].detach()
        pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
        occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
        _, final_traj = self.model.planning(
            cam_front=output['cam_front'].detach(),
            trajs=trajs[:, :, 1:],
            gt_trajs=None,
            cost_volume=output['costvolume'][:, n_present:].detach(),
            semantic_pred=occupancy[:, n_present:].squeeze(2),
            hd_map=output['hdmap'].detach(),
            commands=command,
            target_points=target_points
        )

        steer, throttle, brake, metadata = self.control_pid(final_traj, gt_velocity, tick_data)
        self.pid_metadata = metadata
        self.last_steer = steer

        if brake < 0.05: brake = 0.0
        if throttle > brake: brake = 0.0

        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data, output, final_traj.detach())

        return control

    def save(self, tick_data, output, trajs):
        frame = self.step // 10

        n_present = self.model.receptive_field
        hdmap = output['hdmap'].detach()
        segmentation = output['segmentation'][:, n_present-1].detach()
        pedestrian = output['pedestrian'][:, n_present-1].detach()
        costvolume = output['costvolume'][:, n_present-1].detach()

        import matplotlib
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(16, 8))
        width_ratios = (3,3,3,3)
        gs = matplotlib.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
        gs.update(wspace=0.2, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

        plt.subplot(gs[0,0])
        plt.annotate('FRONT_LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        plt.imshow(tick_data['rgb_left'])
        plt.axis('off')

        plt.subplot(gs[0,1])
        plt.annotate('FRONT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        plt.imshow(tick_data['rgb'])
        plt.axis('off')

        plt.subplot(gs[0,2])
        plt.annotate('FRONT_RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        plt.imshow(tick_data['rgb_right'])
        plt.axis('off')

        plt.subplot(gs[1,1])
        plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        plt.imshow(tick_data['rgb_rear'])
        plt.axis('off')

        plt.subplot(gs[0, 3])
        cost = costvolume[0].cpu().numpy()
        cost = cost[::-1,::-1]
        ax = plt.gca()
        cax = plt.imshow(cost, cmap='viridis')
        cbar = plt.colorbar(cax, extend='both', drawedges = False)
        cbar.set_label('Intensity',size=36, weight =  'bold')
        cbar.ax.tick_params( labelsize=18 )
        cbar.minorticks_on()

        plt.subplot(gs[1, 3])
        showing = torch.zeros((200, 200, 3)).numpy()
        area = torch.argmax(hdmap[0,2:4], dim=0).cpu().numpy()
        hdmap_index = area > 0
        showing[hdmap_index] = np.array([41/255, 255/255, 0/255])

        area = torch.argmax(hdmap[0,0:2], dim=0).cpu().numpy()
        hdmap_index = area > 0
        showing[hdmap_index] = np.array([83/255, 55/255, 122/255])

        semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
        semantic_index = semantic_seg > 0
        showing[semantic_index] = np.array([0/255, 153/255, 255/255])

        pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
        pedestrian_index = pedestrian_seg > 0
        showing[pedestrian_index] = np.array([20/255, 0/255, 255/255])

        plt.imshow(make_contour(showing))
        plt.axis('off')

        bx = np.array([-20, -20])
        dx = np.array([0.2, 0.2])
        w, h = 2.12, 4.90
        pts = np.array([
            [-h/2.+0.5, w/2.],
            [h/2.+0.5, w/2.],
            [h/2.+0.5, -w/2.],
            [-h/2.+0.5, -w/2.],
        ])
        pts = (pts - bx) / dx
        pts[:, [0,1]] = pts[:, [1,0]]
        plt.fill(pts[:, 0], pts[:, 1], '#76b900')

        plt.xlim((200, 0))
        plt.ylim((0, 200))
        trajs[0,:,:1] = trajs[0,:,:1] * -1
        trajs = (trajs[0,:,:2].cpu().numpy() - bx) / dx
        plt.plot(trajs[:, 0], trajs[:, 1])

        plt.annotate('COMMAND:' + str(tick_data['next_command']), (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
        plt.annotate('TARGET_POINT: ({},{})'.format(tick_data['target_point'][0].item(), tick_data['target_point'][1].item()), (0.01, 0.67), c='white', xycoords='axes fraction', fontsize=14 )

        plt.savefig(self.save_path / 'show' / ('%04d.png' % frame))
        plt.close()

        outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
        json.dump(self.pid_metadata, outfile, indent=4)
        outfile.close()