import os
from PIL import Image

import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import quaternion_yaw
from stp3.utils.tools import ( gen_dx_bx, get_nusc_maps)

from stp3.utils.geometry import (
    resize_and_crop_image,
    update_intrinsics,
    calculate_birds_eye_view_parameters,
    convert_egopose_to_matrix_numpy,
    pose_vec2mat,
    mat2pose_vec,
    invert_matrix_egopose_numpy,
    get_global_pose
)
from stp3.utils.instance import convert_instance_mask_to_center_and_offset_label
import stp3.utils.sampler as trajectory_sampler

def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i == len(utimes) or (i > 0 and utime - utimes[i-1] < utimes[i] - utime):
        i -= 1
    return i

class FuturePredictionDataset(torch.utils.data.Dataset):
    SAMPLE_INTERVAL = 0.5 #SECOND
    def __init__(self, nusc, is_train, cfg):
        self.nusc = nusc
        self.dataroot = self.nusc.dataroot
        self.nusc_exp = NuScenesExplorer(nusc)
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
        self.is_train = is_train
        self.cfg = cfg

        if self.is_train == 0:
            self.mode = 'train'
        elif self.is_train == 1:
            self.mode = 'val'
        elif self.is_train == 2:
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.sequence_length = cfg.TIME_RECEPTIVE_FIELD + cfg.N_FUTURE_FRAMES
        self.receptive_field = cfg.TIME_RECEPTIVE_FIELD

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()
        self.indices = self.get_indices()

        # Image resizing and cropping
        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Bird's-eye view parameters
        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )
        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
        )

        # Spatial extent in bird's-eye view, in meters
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # The number of sampled trajectories
        self.n_samples = self.cfg.PLANNING.SAMPLE_NUM

        # HD-map feature extractor
        self.nusc_maps = get_nusc_maps(self.cfg.DATASET.MAP_FOLDER)
        self.scene2map = {}
        for sce in self.nusc.scene:
            log = self.nusc.get('log', sce['log_token'])
            self.scene2map[sce['name']] = log['location']
        self.save_dir = cfg.DATASET.SAVE_DIR

    def get_scenes(self):
        # filter by scene split
        split = {'v1.0-trainval': {0: 'train', 1: 'val', 2: 'test'},
                 'v1.0-mini': {0: 'mini_train', 1: 'mini_val'},}[
            self.nusc.version
        ][self.is_train]

        blacklist = [419] + self.nusc_can.can_blacklist  # # scene-0419 does not have vehicle monitor data
        blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        scenes = create_splits_scenes()[split][:]
        for scene_no in blacklist:
            if scene_no in scenes:
                scenes.remove(scene_no)

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_indices(self):
        indices = []
        for index in range(len(self.ixes)):
            is_valid_data = True
            previous_rec = None
            current_indices = []
            for t in range(self.sequence_length):
                index_t = index + t
                # Going over the dataset size limit.
                if index_t >= len(self.ixes):
                    is_valid_data = False
                    break
                rec = self.ixes[index_t]
                # Check if scene is the same
                if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                    is_valid_data = False
                    break

                current_indices.append(index_t)
                previous_rec = rec

            if is_valid_data:
                indices.append(current_indices)

        return np.asarray(indices)

    def get_resizing_and_cropping_parameters(self):
        original_height, original_width = self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH
        final_height, final_width = self.cfg.IMAGE.FINAL_DIM

        resize_scale = self.cfg.IMAGE.RESIZE_SCALE
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = self.cfg.IMAGE.TOP_CROP
        crop_w = int(max(0, (resized_width - final_width) / 2))
        # Left, top, right, bottom crops.
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        if resized_width != final_width:
            print('Zero padding left and right parts of the image.')
        if crop_h + final_height != resized_height:
            print('Zero padding bottom part of the image.')

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop,
                }

    def get_input_data(self, rec):
        """
        Parameters
        ----------
            rec: nuscenes identifier for a given timestamp

        Returns
        -------
            images: torch.Tensor<float> (N, 3, H, W)
            intrinsics: torch.Tensor<float> (3, 3)
            extrinsics: torch.Tensor(N, 4, 4)
        """
        images = []
        intrinsics = []
        extrinsics = []
        depths = []
        cameras = self.cfg.IMAGE.NAMES

        # The extrinsics we want are from the camera sensor to "flat egopose" as defined
        # https://github.com/nutonomy/nuscenes-devkit/blob/9b492f76df22943daf1dc991358d3d606314af27/python-sdk/nuscenes/nuscenes.py#L279
        # which corresponds to the position of the lidar.
        # This is because the labels are generated by projecting the 3D bounding box in this lidar's reference frame.

        # From lidar egopose to world.
        lidar_sample = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        lidar_pose = self.nusc.get('ego_pose', lidar_sample['ego_pose_token'])
        yaw = Quaternion(lidar_pose['rotation']).yaw_pitch_roll[0]
        lidar_rotation = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)])
        lidar_translation = np.array(lidar_pose['translation'])[:, None]
        lidar_to_world = np.vstack([
            np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
            np.array([0, 0, 0, 1])
        ])

        for cam in cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])

            # Transformation from world to egopose
            car_egopose = self.nusc.get('ego_pose', camera_sample['ego_pose_token'])
            egopose_rotation = Quaternion(car_egopose['rotation']).inverse
            egopose_translation = -np.array(car_egopose['translation'])[:, None]
            world_to_car_egopose = np.vstack([
                np.hstack((egopose_rotation.rotation_matrix, egopose_rotation.rotation_matrix @ egopose_translation)),
                np.array([0, 0, 0, 1])
            ])

            # From egopose to sensor
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])
            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            car_egopose_to_sensor = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            car_egopose_to_sensor = np.linalg.inv(car_egopose_to_sensor)

            # Combine all the transformation.
            # From sensor to lidar.
            lidar_to_sensor = car_egopose_to_sensor @ world_to_car_egopose @ lidar_to_world
            sensor_to_lidar = torch.from_numpy(np.linalg.inv(lidar_to_sensor)).float()

            # Load image
            image_filename = os.path.join(self.dataroot, camera_sample['filename'])
            img = Image.open(image_filename)
            # Resize and crop
            img = resize_and_crop_image(
                img, resize_dims=self.augmentation_parameters['resize_dims'], crop=self.augmentation_parameters['crop']
            )
            # Normalise image
            normalised_img = self.normalise_image(img)

            # Combine resize/cropping in the intrinsics
            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]
            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            # Get Depth
            # Depth data should under the dataroot path 
            if self.cfg.LIFT.GT_DEPTH:
                base_root = os.path.join(self.dataroot, 'depths') 
                filename = os.path.basename(camera_sample['filename']).split('.')[0] + '.npy'
                depth_file_name = os.path.join(base_root, cam, 'npy', filename)
                depth = torch.from_numpy(np.load(depth_file_name)).unsqueeze(0).unsqueeze(0)
                depth = F.interpolate(depth, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear')
                depth = depth.squeeze()
                crop = self.augmentation_parameters['crop']
                depth = depth[crop[1]:crop[3], crop[0]:crop[2]]
                depth = torch.round(depth)
                depths.append(depth.unsqueeze(0).unsqueeze(0))

            images.append(normalised_img.unsqueeze(0).unsqueeze(0))
            intrinsics.append(intrinsic.unsqueeze(0).unsqueeze(0))
            extrinsics.append(sensor_to_lidar.unsqueeze(0).unsqueeze(0))

        images, intrinsics, extrinsics = (torch.cat(images, dim=1),
                                          torch.cat(intrinsics, dim=1),
                                          torch.cat(extrinsics, dim=1)
                                          )
        if len(depths) > 0:
            depths = torch.cat(depths, dim=1)

        return images, intrinsics, extrinsics, depths

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_depth_from_lidar(self, lidar_sample, cam_sample):
        points, coloring, im = self.nusc_exp.map_pointcloud_to_image(lidar_sample, cam_sample)
        tmp_cam = np.zeros((self.cfg.IMAGE.ORIGINAL_HEIGHT, self.cfg.IMAGE.ORIGINAL_WIDTH))
        points = points.astype(np.int)
        tmp_cam[points[1, :], points[0,:]] = coloring
        tmp_cam = torch.from_numpy(tmp_cam).unsqueeze(0).unsqueeze(0)
        tmp_cam = F.interpolate(tmp_cam, scale_factor=self.cfg.IMAGE.RESIZE_SCALE, mode='bilinear', align_corners=False, recompute_scale_factor=True)
        tmp_cam = tmp_cam.squeeze()
        crop = self.augmentation_parameters['crop']
        tmp_cam = tmp_cam[crop[1]:crop[3], crop[0]:crop[2]]
        tmp_cam = torch.round(tmp_cam)
        return tmp_cam


    def get_birds_eye_view_label(self, rec, instance_map, in_pred):
        translation, rotation = self._get_top_lidar_pose(rec)
        segmentation = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))
        # Background is ID 0
        instance = np.zeros((self.bev_dimension[0], self.bev_dimension[1]))

        for annotation_token in rec['anns']:
            # Filter out all non vehicle instances
            annotation = self.nusc.get('sample_annotation', annotation_token)

            if self.cfg.DATASET.FILTER_INVISIBLE_VEHICLES and int(annotation['visibility_token']) == 1 and in_pred is False:
                continue
            if in_pred is True and annotation['instance_token'] not in instance_map:
                continue

            # NuScenes filter
            if 'vehicle' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                instance_id = instance_map[annotation['instance_token']]
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(instance, [poly_region], instance_id)
                cv2.fillPoly(segmentation, [poly_region], 1.0)
            elif 'human' in annotation['category_name']:
                if annotation['instance_token'] not in instance_map:
                    instance_map[annotation['instance_token']] = len(instance_map) + 1
                poly_region, z = self._get_poly_region_in_image(annotation, translation, rotation)
                cv2.fillPoly(pedestrian, [poly_region], 1.0)


        return segmentation, instance, pedestrian, instance_map

    def _get_poly_region_in_image(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round((pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]
        return pts, z

    def get_label(self, rec, instance_map, in_pred):
        segmentation_np, instance_np, pedestrian_np, instance_map = \
            self.get_birds_eye_view_label(rec, instance_map, in_pred)
        segmentation = torch.from_numpy(segmentation_np).long().unsqueeze(0).unsqueeze(0)
        instance = torch.from_numpy(instance_np).long().unsqueeze(0)
        pedestrian = torch.from_numpy(pedestrian_np).long().unsqueeze(0).unsqueeze(0)

        return segmentation, instance, pedestrian, instance_map

    def get_future_egomotion(self, rec, index):
        rec_t0 = rec

        # Identity
        future_egomotion = np.eye(4, dtype=np.float32)

        if index < len(self.ixes) - 1:
            rec_t1 = self.ixes[index + 1]

            if rec_t0['scene_token'] == rec_t1['scene_token']:
                egopose_t0 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t0['data']['LIDAR_TOP'])['ego_pose_token']
                )
                egopose_t1 = self.nusc.get(
                    'ego_pose', self.nusc.get('sample_data', rec_t1['data']['LIDAR_TOP'])['ego_pose_token']
                )

                egopose_t0 = convert_egopose_to_matrix_numpy(egopose_t0)
                egopose_t1 = convert_egopose_to_matrix_numpy(egopose_t1)

                future_egomotion = invert_matrix_egopose_numpy(egopose_t1).dot(egopose_t0)
                future_egomotion[3, :3] = 0.0
                future_egomotion[3, 3] = 1.0

        future_egomotion = torch.Tensor(future_egomotion).float()

        # Convert to 6DoF vector
        future_egomotion = mat2pose_vec(future_egomotion)
        return future_egomotion.unsqueeze(0)

    def get_trajectory_sampling(self, rec=None, sample_indice=None):
        if rec is None and sample_indice is None:
            raise ValueError("No valid input rec or token")
        if rec is None and sample_indice is not None:
            rec = self.ixes[sample_indice]

        ref_scene = self.nusc.get("scene", rec['scene_token'])

        # vm_msgs = self.nusc_can.get_messages(ref_scene['name'], 'vehicle_monitor')
        # vm_uts = [msg['utime'] for msg in vm_msgs]
        pose_msgs = self.nusc_can.get_messages(ref_scene['name'],'pose')
        pose_uts = [msg['utime'] for msg in pose_msgs]
        steer_msgs = self.nusc_can.get_messages(ref_scene['name'], 'steeranglefeedback')
        steer_uts = [msg['utime'] for msg in steer_msgs]

        ref_utime = rec['timestamp']
        # vm_index = locate_message(vm_uts, ref_utime)
        # vm_data = vm_msgs[vm_index]
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]

        # initial speed
        # v0 = vm_data["vehicle_speed"] / 3.6  # km/h to m/s
        v0 = pose_data["vel"][0]  # [0] means longitudinal velocity  m/s

        # curvature (positive: turn left)
        # steering = np.deg2rad(vm_data["steering"])
        steering = steer_data["value"]

        location = self.scene2map[ref_scene['name']]
        # flip x axis if in left-hand traffic (singapore)
        flip_flag = True if location.startswith('singapore') else False
        if flip_flag:
            steering *= -1
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

    def voxelize_hd_map(self, rec):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        stretch = [self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1]]
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.scene2map[self.nusc.get('scene', rec['scene_token'])['name']]

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1,0], rot[0,0]) # in radian
        center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

        box_coords = (
            center[0],
            center[1],
            stretch[0]*2,
            stretch[1]*2
        ) # (x_center, y_center, width, height)
        canvas_size = (
                int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
                int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        elements = self.cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
        hd_features = self.nusc_maps[map_name].get_map_mask(box_coords, rot * 180 / np.pi , elements, canvas_size=canvas_size)
        #traffic = self.hd_traffic_light(map_name, center, stretch, dx, bx, canvas_size)
        #return torch.from_numpy(np.concatenate((hd_features, traffic), axis=0)[None]).float()
        hd_features = torch.from_numpy(hd_features[None]).float()
        hd_features = torch.transpose(hd_features,-2,-1) # (y,x) replace horizontal and vertical coordinates
        return hd_features

    def hd_traffic_light(self, map_name, center, stretch, dx, bx, canvas_size):

        roads = np.zeros(canvas_size)
        my_patch = (
            center[0] - stretch[0],
            center[1] - stretch[1],
            center[0] + stretch[0],
            center[1] + stretch[1],
        )
        tl_token = self.nusc_maps[map_name].get_records_in_patch(my_patch, ['traffic_light'], mode='intersect')['traffic_light']
        polys = []
        for token in tl_token:
            road_token =self.nusc_maps[map_name].get('traffic_light', token)['from_road_block_token']
            pt = self.nusc_maps[map_name].get('road_block', road_token)['polygon_token']
            polygon = self.nusc_maps[map_name].extract_polygon(pt)
            polys.append(np.array(polygon.exterior.xy).T)

        def get_rot(h):
            return torch.Tensor([
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ])
        # convert to local coordinates in place
        rot = get_rot(np.arctan2(center[3], center[2])).T
        for rowi in range(len(polys)):
            polys[rowi] -= center[:2]
            polys[rowi] = np.dot(polys[rowi], rot)

        for la in polys:
            pts = (la - bx) / dx
            pts = np.int32(np.around(pts))
            cv2.fillPoly(roads, [pts], 1)

        return roads[None]

    def get_gt_trajectory(self, rec, ref_index):
        n_output = self.cfg.N_FUTURE_FRAMES
        gt_trajectory = np.zeros((n_output+1, 3), np.float64)

        egopose_cur = get_global_pose(rec, self.nusc, inverse=True)

        for i in range(n_output+1):
            index = ref_index + i
            if index < len(self.ixes):
                rec_future = self.ixes[index]

                egopose_future = get_global_pose(rec_future, self.nusc, inverse=False)

                egopose_future = egopose_cur.dot(egopose_future)
                theta = quaternion_yaw(Quaternion(matrix=egopose_future))

                origin = np.array(egopose_future[:3, 3])

                gt_trajectory[i, :] = [origin[0], origin[1], theta]

        if gt_trajectory[-1][0] >= 2:
            command = 'RIGHT'
        elif gt_trajectory[-1][0] <= -2:
            command = 'LEFT'
        else:
            command = 'FORWARD'

        return gt_trajectory, command

    def get_routed_map(self, gt_points):
        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2].numpy(), bx[:2].numpy()

        canvas_size = (
            int(self.cfg.LIFT.X_BOUND[1] * 2 / self.cfg.LIFT.X_BOUND[2]),
            int(self.cfg.LIFT.Y_BOUND[1] * 2 / self.cfg.LIFT.Y_BOUND[2])
        )

        roads = np.zeros(canvas_size)
        W = 1.85
        pts = np.array([
            [-4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, W / 2.],
            [4.084 / 2. + 0.5, -W / 2.],
            [-4.084 / 2. + 0.5, -W / 2.],
        ])
        pts = (pts - bx) / dx
        pts[:, [0, 1]] = pts[:, [1, 0]]

        pts = np.int32(np.around(pts))
        cv2.fillPoly(roads, [pts], 1)

        gt_points = gt_points[:-1].numpy()
        # 坐标原点在左上角
        target = pts.copy()
        target[:,0] = pts[:,0] + gt_points[0] / dx[0]
        target[:,1] = pts[:,1] - gt_points[1] / dx[1]
        target = np.int32(np.around(target))
        cv2.fillPoly(roads, [target], 1)
        return roads

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns
        -------
            data: dict with the following keys:
                image: torch.Tensor<float> (T, N, 3, H, W)
                    normalised cameras images with T the sequence length, and N the number of cameras.
                intrinsics: torch.Tensor<float> (T, N, 3, 3)
                    intrinsics containing resizing and cropping parameters.
                extrinsics: torch.Tensor<float> (T, N, 4, 4)
                    6 DoF pose from world coordinates to camera coordinates.
                segmentation: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                    (H_bev, W_bev) are the pixel dimensions in bird's-eye view.
                instance: torch.Tensor<int64> (T, 1, H_bev, W_bev)
                centerness: torch.Tensor<float> (T, 1, H_bev, W_bev)
                offset: torch.Tensor<float> (T, 2, H_bev, W_bev)
                flow: torch.Tensor<float> (T, 2, H_bev, W_bev)
                future_egomotion: torch.Tensor<float> (T, 6)
                    6 DoF egomotion t -> t+1

        """
        data = {}
        keys = ['image', 'intrinsics', 'extrinsics', 'depths',
                'segmentation', 'instance', 'centerness', 'offset', 'flow', 'pedestrian',
                'future_egomotion', 'hdmap', 'gt_trajectory', 'indices',
                ]
        for key in keys:
            data[key] = []

        instance_map = {}
        # Loop over all the frames in the sequence.
        for i, index_t in enumerate(self.indices[index]):
            if i >= self.receptive_field:
                in_pred = True
            else:
                in_pred = False
            rec = self.ixes[index_t]

            if i < self.receptive_field:
                images, intrinsics, extrinsics, depths = self.get_input_data(rec)
                data['image'].append(images)
                data['intrinsics'].append(intrinsics)
                data['extrinsics'].append(extrinsics)
                data['depths'].append(depths)
            segmentation, instance, pedestrian, instance_map = self.get_label(rec, instance_map, in_pred)

            future_egomotion = self.get_future_egomotion(rec, index_t)
            hd_map_feature = self.voxelize_hd_map(rec)

            data['segmentation'].append(segmentation)
            data['instance'].append(instance)
            data['pedestrian'].append(pedestrian)
            data['future_egomotion'].append(future_egomotion)
            data['hdmap'].append(hd_map_feature)
            data['indices'].append(index_t)

            if i == self.cfg.TIME_RECEPTIVE_FIELD-1:
                gt_trajectory, command = self.get_gt_trajectory(rec, index_t)
                data['gt_trajectory'] = torch.from_numpy(gt_trajectory).float()
                data['command'] = command
                trajs = self.get_trajectory_sampling(rec)
                data['sample_trajectory'] = torch.from_numpy(trajs).float()

        for key, value in data.items():
            if key in ['image', 'intrinsics', 'extrinsics', 'depths', 'segmentation', 'instance', 'future_egomotion', 'hdmap', 'pedestrian']:
                if key == 'depths' and self.cfg.LIFT.GT_DEPTH is False:
                    continue
                data[key] = torch.cat(value, dim=0)

        data['target_point'] = torch.tensor([0., 0.])
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label(
            data['instance'], data['future_egomotion'],
            num_instances=len(instance_map), ignore_index=self.cfg.DATASET.IGNORE_INDEX, subtract_egomotion=True,
            spatial_extent=self.spatial_extent,
        )
        data['centerness'] = instance_centerness
        data['offset'] = instance_offset
        data['flow'] = instance_flow
        return data