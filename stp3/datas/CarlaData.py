import os
import json
import numpy as np
import cv2
import torch
import torch.utils.data
import torchvision
from PIL import Image
import PIL
from pyquaternion import Quaternion

from stp3.utils.geometry import (
    update_intrinsics,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
)
import stp3.utils.sampler as trajectory_sampler


class CarlaDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5  # SECOND
    def __init__(self, root_dir, is_train, cfg):
        super(CarlaDataset, self).__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD
        self.cfg = cfg
        self.n_samples = self.cfg.PLANNING.SAMPLE_NUM

        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ]
        )

        self.front = []
        self.left = []
        self.right = []
        self.rear = []
        self.front_depth = []
        self.left_depth = []
        self.right_depth = []
        self.rear_depth = []
        self.topdown = []
        self.x = []
        self.y = []
        self.x_command = []
        self.y_command = []
        self.theta = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.command = []
        self.velocity = []
        self.hdmap = []

        self.get_train_val()

    def get_train_val(self):
        train_towns = ['Town01', 'Town02', 'Town03', 'Town04', 'Town06', 'Town10']
        val_towns = ['Town05']
        train_data, val_data = [], []
        for town in train_towns:
            train_data.append(os.path.join(self.root_dir, town+'_tiny'))
            train_data.append(os.path.join(self.root_dir, town+'_short'))
        for town in val_towns:
            val_data.append(os.path.join(self.root_dir, town+'_short'))

        require_data = train_data if self.is_train else val_data

        for subroot in require_data:
            preload_file = os.path.join(subroot, 'cam_mea_topdown_'+str(self.receptive_field)+'_'+str(self.sequence_length)+'.npy')

            if not os.path.exists(preload_file):
                preload_front = []
                preload_left = []
                preload_right = []
                preload_rear = []
                preload_front_depth = []
                preload_left_depth = []
                preload_right_depth = []
                preload_rear_depth = []
                preload_topdown = []
                preload_x = []
                preload_y = []
                preload_x_command = []
                preload_y_command = []
                preload_theta = []
                preload_steer = []
                preload_throttle = []
                preload_brake = []
                preload_command = []
                preload_velocity = []
                preload_hdmap = []

                root_files = os.listdir(subroot)
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(subroot, folder))]
                for route in routes:
                    route_dir = os.path.join(subroot, route)
                    num_seq = len(os.listdir(route_dir + "/rgb_front/")) - self.sequence_length
                    for seq in range(num_seq):
                        fronts, lefts, rights, rears = [], [], [], []
                        fr_depths, le_depths, ri_depths, re_depths = [], [], [], []
                        xs, ys, thetas = [], [], []
                        top_down, hdmap = [], []
                        for i in range(self.receptive_field):
                            filename = f"{str(seq+1+i).zfill(4)}.png"
                            fronts.append(route_dir + "/rgb_front/" + filename)
                            lefts.append(route_dir + "/rgb_left/" + filename)
                            rights.append(route_dir + "/rgb_right/" + filename)
                            rears.append(route_dir + "/rgb_rear/" + filename)
                            fr_depths.append(route_dir + "/depth_front/" + filename)
                            le_depths.append(route_dir + "/depth_left/" + filename)
                            ri_depths.append(route_dir + "/depth_right/" + filename)
                            re_depths.append(route_dir + "/depth_rear/" + filename)
                            top_down.append(route_dir + "/topdown/" + filename)
                            hdmap.append(route_dir + "/hdmap/" + filename)
                            # position
                            with open(route_dir + f"/measurements/{str(seq+1+i).zfill(4)}.json","r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            thetas.append(data['theta'])

                        preload_x_command.append(data['x_command'])
                        preload_y_command.append(data['y_command'])
                        preload_steer.append(data['steer'])
                        preload_throttle.append(data['throttle'])
                        preload_brake.append(data['brake'])
                        preload_command.append(data['command'])
                        preload_velocity.append(data['speed'])

                        for i in range(self.receptive_field, self.sequence_length):
                            filename = f"{str(seq + 1 + i).zfill(4)}.png"
                            top_down.append(route_dir + "/topdown/" + filename)

                            with open(route_dir + f"/measurements/{str(seq+1+i).zfill(4)}.json","r") as read_file:
                                data = json.load(read_file)
                            xs.append(data['x'])
                            ys.append(data['y'])
                            if np.isnan(data['theta']):
                                thetas.append(0)
                            else:
                                thetas.append(data['theta'])

                        preload_front.append(fronts)
                        preload_left.append(lefts)
                        preload_right.append(rights)
                        preload_rear.append(rears)
                        preload_front_depth.append(fr_depths)
                        preload_left_depth.append(le_depths)
                        preload_right_depth.append(ri_depths)
                        preload_rear_depth.append(re_depths)
                        preload_x.append(xs)
                        preload_y.append(ys)
                        preload_theta.append(thetas)
                        preload_topdown.append(top_down)
                        preload_hdmap.append(hdmap)

                preload_dict = {}
                preload_dict['front'] = preload_front
                preload_dict['left'] = preload_left
                preload_dict['right'] = preload_right
                preload_dict['rear'] = preload_rear
                preload_dict['front_depth'] = preload_front_depth
                preload_dict['left_depth'] = preload_left_depth
                preload_dict['right_depth'] = preload_right_depth
                preload_dict['rear_depth'] = preload_rear_depth
                preload_dict['topdown'] = preload_topdown
                preload_dict['x'] = preload_x
                preload_dict['y'] = preload_y
                preload_dict['x_command'] = preload_x_command
                preload_dict['y_command'] = preload_y_command
                preload_dict['theta'] = preload_theta
                preload_dict['steer'] = preload_steer
                preload_dict['throttle'] = preload_throttle
                preload_dict['brake'] = preload_brake
                preload_dict['command'] = preload_command
                preload_dict['velocity'] = preload_velocity
                preload_dict['hdmap'] = preload_hdmap
                np.save(preload_file, preload_dict)

            # load from npy if available
            preload_dict = np.load(preload_file, allow_pickle=True)
            self.front += preload_dict.item()['front']
            self.left += preload_dict.item()['left']
            self.right += preload_dict.item()['right']
            self.rear += preload_dict.item()['rear']
            self.front_depth += preload_dict.item()['front_depth']
            self.left_depth += preload_dict.item()['left_depth']
            self.right_depth += preload_dict.item()['right_depth']
            self.rear_depth += preload_dict.item()['rear_depth']
            self.topdown += preload_dict.item()['topdown']
            self.x += preload_dict.item()['x']
            self.y += preload_dict.item()['y']
            self.x_command += preload_dict.item()['x_command']
            self.y_command += preload_dict.item()['y_command']
            self.theta += preload_dict.item()['theta']
            self.steer += preload_dict.item()['steer']
            self.throttle += preload_dict.item()['throttle']
            self.brake += preload_dict.item()['brake']
            self.command += preload_dict.item()['command']
            self.velocity += preload_dict.item()['velocity']
            self.hdmap += preload_dict.item()['hdmap']
            print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        return len(self.front)

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

    def get_hdmap(self, path, scale, crop):
        image = Image.open(path)
        (width, height) = (int(image.width // scale), int(image.height // scale))
        im_resized = image.resize((width, height))
        image = np.asarray(im_resized)
        start_x = height // 2 - crop // 2
        start_y = width // 2 - crop // 2
        cropped_image = image[start_x:start_x + crop, start_y:start_y + crop]
        h, w, c = cropped_image.shape
        lane_index = np.all((cropped_image == [255, 0, 255]), axis=2)
        lane = np.zeros((h, w))
        lane[lane_index] = 1
        drivable_index = np.all((cropped_image == [54, 52, 46]), axis=2)
        drivable = np.zeros((h, w))
        drivable[drivable_index] = 1
        drivable = np.logical_or(drivable, lane)
        # down, right is the positive
        lane = lane[::-1,::-1]
        drivable = drivable[::-1,::-1]
        hdmap = np.concatenate([lane[None], drivable[None]], axis=0)
        return hdmap

    def get_labels(self, path, scale, crop):
        image = Image.open(path)
        (width, height) = (int(image.width // scale), int(image.height // scale))
        im_resized = image.resize((width, height), resample=PIL.Image.NEAREST)
        image = np.asarray(im_resized)
        start_x = height // 2 - crop // 2
        start_y = width // 2 - crop // 2
        cropped_image = image[start_x:start_x + crop, start_y:start_y + crop]
        h, w = cropped_image.shape
        vehicle_index = cropped_image == 10
        vehicle = np.zeros((h, w))
        vehicle[vehicle_index] = 1
        vehicle[89:112,96:105] = 0
        pedestrian_index = cropped_image == 4
        pedestrian = np.zeros((h, w))
        pedestrian[pedestrian_index] = 1
        vehicle = vehicle[::-1,::-1]
        pedestrian = pedestrian[::-1,::-1]
        return vehicle.copy(), pedestrian.copy()

    def get_trajectory_sampling(self, v0, steering):

        Kappa = 2 * steering / 2.588

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])  # define side

        t_start = 0  # second
        t_end = self.cfg.N_FUTURE_FRAMES * self.SAMPLE_INTERVAL  # second
        t_interval = self.SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(v0, Kappa, T0, N0, tt, self.n_samples)
        sampled_trajectories = sampled_trajectories_fine[:, ::10]
        return sampled_trajectories

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

    def get_depth(self, data):
        """
        Computes the normalized depth
        """
        data = data.astype(np.float32)

        normalized = np.dot(data, [65536.0, 256.0, 1.0])
        normalized /= (256 * 256 * 256 - 1)
        return torch.from_numpy(normalized * 1000)

    def __getitem__(self, index):
        data = {}
        keys = ['image', 'depths', 'segmentation', 'pedestrian', 'extrinsics', 'intrinsics', 'hdmap', 'gt_trajectory']
        for key in keys:
            data[key] = []

        seq_fronts = self.front[index]
        seq_lefts = self.left[index]
        seq_rights = self.right[index]
        seq_rears = self.rear[index]
        seq_front_depths = self.front_depth[index]
        seq_left_depths = self.left_depth[index]
        seq_right_depths = self.right_depth[index]
        seq_rear_depths = self.rear_depth[index]
        seq_hdmaps = self.hdmap[index]
        seq_topdowns = self.topdown[index]
        seq_x = self.x[index]
        seq_y = self.y[index]
        seq_theta = self.theta[index]

        for i in range(self.receptive_field):
            images = []
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_fronts[i]), scale=1., crop=256))).unsqueeze(0))
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_lefts[i]), scale=1., crop=256))).unsqueeze(0))
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_rights[i]), scale=1., crop=256))).unsqueeze(0))
            images.append(self.normalise_image(np.array(
                scale_and_crop_image(Image.open(seq_rears[i]), scale=1.,crop=256))).unsqueeze(0))
            images = torch.cat(images, dim=0)
            data['image'].append(images.unsqueeze(0))
            depths = []
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_front_depths[i]), scale=1., crop=256))).unsqueeze(0))
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_left_depths[i]), scale=1., crop=256))).unsqueeze(0))
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_right_depths[i]), scale=1., crop=256))).unsqueeze(0))
            depths.append(self.get_depth(np.array(
                scale_and_crop_image(Image.open(seq_rear_depths[i]), scale=1., crop=256))).unsqueeze(0))
            depths = torch.cat(depths, dim=0)
            data['depths'].append(depths.unsqueeze(0))
            extrinsics, intrinsics = self.get_cam_para()
            data['extrinsics'].append(extrinsics.unsqueeze(0))
            data['intrinsics'].append(intrinsics.unsqueeze(0))
            data['hdmap'].append(torch.from_numpy(self.get_hdmap(seq_hdmaps[i], 1., 200)).unsqueeze(0))


            # fix for theta=nan in some measurements
            if np.isnan(seq_theta[i]):
                seq_theta[i] = 0.
        ego_x = seq_x[self.receptive_field-1]
        ego_y = seq_y[self.receptive_field-1]
        ego_theta = seq_theta[self.receptive_field-1]

        for i in range(self.sequence_length):
            if i >= self.receptive_field-1:
                local_waypoint = transform_2d_points(np.zeros((1, 3)),
                                                     np.pi / 2 - seq_theta[i], -seq_x[i], -seq_y[i],
                                                     np.pi / 2 - ego_theta, -ego_x, -ego_y)
                local_waypoint = local_waypoint * [1.0, -1.0, 1.0]
                data['gt_trajectory'].append(torch.from_numpy(local_waypoint))
            segmentation, pedestrian = self.get_labels(seq_topdowns[i], 1.1, 200)
            data['segmentation'].append(torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0))
            data['pedestrian'].append(torch.from_numpy(pedestrian).unsqueeze(0).unsqueeze(0))

        R = np.array([
            [np.cos(np.pi / 2 + ego_theta), -np.sin(np.pi / 2 + ego_theta)],
            [np.sin(np.pi / 2 + ego_theta), np.cos(np.pi / 2 + ego_theta)]
        ])
        local_command_point = np.array([self.x_command[index] - ego_x, self.y_command[index] - ego_y])
        local_command_point = R.T.dot(local_command_point)
        local_command_point = local_command_point * [1.0, -1.0]
        data['target_point'] = torch.from_numpy(local_command_point)

        if self.command[index] == 1:
            data['command'] = 'LEFT'
        elif self.command[index] == 2:
            data['command'] = 'RIGHT'
        elif self.command[index] == 3:
            data['command'] = 'FORWARD'
        else:
            data['command'] = 'LANE'
        data['steer'] = self.steer[index]
        data['throttle'] = self.throttle[index]
        data['brake'] = self.brake[index]
        data['velocity'] = self.velocity[index]
        data['future_egomotion'] = self.get_future_egomotion(seq_x, seq_y, seq_theta)
        data['sample_trajectory'] = torch.from_numpy(self.get_trajectory_sampling(self.velocity[index], self.steer[index])).float()

        for key, value in data.items():
            if key in keys:
                data[key] = torch.cat(value, dim=0)

        return data



def scale_and_crop_image(image, scale=1., crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height//2 - crop//2
    start_y = width//2 - crop//2
    cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    return cropped_image


def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
    """
    Build a rotation matrix and take the dot product.
    """
    # z value to 1 for rotation
    xy1 = xyz.copy()
    xy1[:, 2] = 1

    c, s = np.cos(r1), np.sin(r1)
    r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

    # np.dot converts to a matrix, so we explicitly change it back to an array
    world = np.asarray(r1_to_world @ xy1.T)

    c, s = np.cos(r2), np.sin(r2)
    r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
    world_to_r2 = np.linalg.inv(r2_to_world)

    out = np.asarray(world_to_r2 @ world).T

    # reset z-coordinate
    out[:, 2] = xyz[:, 2]

    return out
