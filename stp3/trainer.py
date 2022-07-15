import torch
import torch.nn as nn
import pytorch_lightning as pl

from stp3.config import get_cfg
from stp3.models.stp3 import STP3
from stp3.losses import SpatialRegressionLoss, SegmentationLoss, HDmapLoss, DepthLoss
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.geometry import cumulative_warp_features_reverse, cumulative_warp_features
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import visualise_output


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        # see config.py for details
        self.hparams = hparams
        # pytorch lightning does not support saving YACS CfgNone
        cfg = get_cfg(cfg_dict=self.hparams)
        self.cfg = cfg
        self.n_classes = len(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
        self.hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS

        # Bird's-eye view extent in meters
        assert self.cfg.LIFT.X_BOUND[1] > 0 and self.cfg.LIFT.Y_BOUND[1] > 0
        self.spatial_extent = (self.cfg.LIFT.X_BOUND[1], self.cfg.LIFT.Y_BOUND[1])

        # Model
        self.model = STP3(cfg)

        self.losses_fn = nn.ModuleDict()

        # Semantic segmentation
        self.losses_fn['segmentation'] = SegmentationLoss(
            class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS),
            use_top_k=self.cfg.SEMANTIC_SEG.VEHICLE.USE_TOP_K,
            top_k_ratio=self.cfg.SEMANTIC_SEG.VEHICLE.TOP_K_RATIO,
            future_discount=self.cfg.FUTURE_DISCOUNT,
        )
        self.model.segmentation_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.metric_vehicle_val = IntersectionOverUnion(self.n_classes)

        # Pedestrian segmentation
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            self.losses_fn['pedestrian'] = SegmentationLoss(
                class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.PEDESTRIAN.WEIGHTS),
                use_top_k=self.cfg.SEMANTIC_SEG.PEDESTRIAN.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.PEDESTRIAN.TOP_K_RATIO,
                future_discount=self.cfg.FUTURE_DISCOUNT,
            )
            self.model.pedestrian_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_pedestrian_val = IntersectionOverUnion(self.n_classes)

        # HD map
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            self.losses_fn['hdmap'] = HDmapLoss(
                class_weights=torch.Tensor(self.cfg.SEMANTIC_SEG.HDMAP.WEIGHTS),
                training_weights=self.cfg.SEMANTIC_SEG.HDMAP.TRAIN_WEIGHT,
                use_top_k=self.cfg.SEMANTIC_SEG.HDMAP.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.HDMAP.TOP_K_RATIO,
            )
            self.metric_hdmap_val = []
            for i in range(len(self.hdmap_class)):
                self.metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1))
            self.model.hdmap_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_hdmap_val = nn.ModuleList(self.metric_hdmap_val)

        # Depth
        if self.cfg.LIFT.GT_DEPTH:
            self.losses_fn['depths'] = DepthLoss()
            self.model.depths_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Instance segmentation
        if self.cfg.INSTANCE_SEG.ENABLED:
            self.losses_fn['instance_center'] = SpatialRegressionLoss(
                norm=2, future_discount=self.cfg.FUTURE_DISCOUNT
            )
            self.losses_fn['instance_offset'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.centerness_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.model.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.metric_panoptic_val = PanopticMetric(n_classes=self.n_classes)

        # Instance flow
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.losses_fn['instance_flow'] = SpatialRegressionLoss(
                norm=1, future_discount=self.cfg.FUTURE_DISCOUNT, ignore_index=self.cfg.DATASET.IGNORE_INDEX
            )
            self.model.flow_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Planning
        if self.cfg.PLANNING.ENABLED:
            self.metric_planning_val = PlanningMetric(self.cfg, self.cfg.N_FUTURE_FRAMES)
            self.model.planning_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.training_step_count = 0

    def shared_step(self, batch, is_train):
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        B = len(image)

        # Warp labels
        labels = self.prepare_future_labels(batch)

        # Forward pass
        output = self.model(
            image, intrinsics, extrinsics, future_egomotion,
        )

        #####
        # Loss computation
        #####
        loss = {}
        if is_train:
            # segmentation
            segmentation_factor = 1 / (2 * torch.exp(self.model.segmentation_weight))
            loss['segmentation'] = segmentation_factor * self.losses_fn['segmentation'](
                output['segmentation'], labels['segmentation'], self.model.receptive_field
            )
            loss['segmentation_uncertainty'] = 0.5 * self.model.segmentation_weight

            # Pedestrian
            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                pedestrian_factor = 1 / (2 * torch.exp(self.model.pedestrian_weight))
                loss['pedestrian'] = pedestrian_factor * self.losses_fn['pedestrian'](
                    output['pedestrian'], labels['pedestrian'], self.model.receptive_field
                )
                loss['pedestrian_uncertainty'] = 0.5 * self.model.pedestrian_weight

            # hdmap loss
            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                hdmap_factor = 1 / (2 * torch.exp(self.model.hdmap_weight))
                loss['hdmap'] = hdmap_factor * self.losses_fn['hdmap'](output['hdmap'], labels['hdmap'])
                loss['hdmap_uncertainty'] = 0.5 * self.model.hdmap_weight

            if self.cfg.INSTANCE_SEG.ENABLED:
                # instance center
                centerness_factor = 1 / (2 * torch.exp(self.model.centerness_weight))
                loss['instance_center'] = centerness_factor * self.losses_fn['instance_center'](
                    output['instance_center'], labels['centerness'], self.model.receptive_field
                )
                loss['centerness_uncertainty'] = 0.5 * self.model.centerness_weight

                # instance offset
                offset_factor = 1 / (2 * torch.exp(self.model.offset_weight))
                loss['instance_offset'] = offset_factor * self.losses_fn['instance_offset'](
                    output['instance_offset'], labels['offset'], self.model.receptive_field
                )
                loss['offset_uncertainty'] = 0.5 * self.model.offset_weight

            # depth loss
            if self.cfg.LIFT.GT_DEPTH:
                depths_factor = 1 / (2 * torch.exp(self.model.depths_weight))
                loss['depths'] = depths_factor * self.losses_fn['depths'](output['depth_prediction'], labels['depths'])
                loss['depths_uncertainty'] = 0.5 * self.model.depths_weight

            # instance flow
            if self.cfg.INSTANCE_FLOW.ENABLED:
                flow_factor = 1 / (2 * torch.exp(self.model.flow_weight))
                loss['instance_flow'] = flow_factor * self.losses_fn['instance_flow'](
                    output['instance_flow'], labels['flow'], self.model.receptive_field
                )
                loss['flow_uncertainty'] = 0.5 * self.model.flow_weight

            # Planning
            if self.cfg.PLANNING.ENABLED:
                receptive_field = self.model.receptive_field
                planning_factor = 1 / (2 * torch.exp(self.model.planning_weight))
                occupancy = torch.logical_or(labels['segmentation'][:, receptive_field:].squeeze(2),
                                             labels['pedestrian'][:, receptive_field:].squeeze(2))
                pl_loss, final_traj = self.model.planning(
                    cam_front=output['cam_front'].detach(),
                    trajs=trajs[:, :, 1:],
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    cost_volume=output['costvolume'][:, receptive_field:],
                    semantic_pred=occupancy,
                    hd_map=labels['hdmap'],
                    commands=command,
                    target_points=target_points
                )
                loss['planning'] = planning_factor * pl_loss
                loss['planning_uncertainty'] = 0.5 * self.model.planning_weight
                output = {**output, 'selected_traj': torch.cat(
                    [torch.zeros((B, 1, 3), device=final_traj.device), final_traj], dim=1)}
            else:
                output = {**output, 'selected_traj': labels['gt_trajectory']}

        # Metrics
        else:
            n_present = self.model.receptive_field

            # semantic segmentation metric
            seg_prediction = output['segmentation'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
            self.metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

            # pedestrian segmentation metric
            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                pedestrian_prediction = output['pedestrian'].detach()
                pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
                self.metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                           labels['pedestrian'][:, n_present - 1:])
            else:
                pedestrian_prediction = torch.zeros_like(seg_prediction)

            # hdmap metric
            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                for i in range(len(self.hdmap_class)):
                    hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                    hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                    self.metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

            # instance segmentation metric
            if self.cfg.INSTANCE_SEG.ENABLED:
                pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                    output, compute_matched_centers=False
                )
                self.metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                         labels['instance'][:, n_present - 1:])

            # planning metric
            if self.cfg.PLANNING.ENABLED:
                occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
                _, final_traj = self.model.planning(
                    cam_front=output['cam_front'].detach(),
                    trajs=trajs[:, :, 1:],
                    gt_trajs=labels['gt_trajectory'][:, 1:],
                    cost_volume=output['costvolume'][:, n_present:].detach(),
                    semantic_pred=occupancy[:, n_present:].squeeze(2),
                    hd_map=output['hdmap'].detach(),
                    commands=command,
                    target_points=target_points
                )
                occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                             labels['pedestrian'][:, n_present:].squeeze(2))
                self.metric_planning_val(final_traj, labels['gt_trajectory'][:, 1:], occupancy)
                output = {**output,
                          'selected_traj': torch.cat([torch.zeros((B, 1, 3), device=final_traj.device), final_traj],
                                                     dim=1)}
            else:
                output = {**output, 'selected_traj': labels['gt_trajectory']}

        return output, labels, loss

    def prepare_future_labels(self, batch):
        labels = {}

        segmentation_labels = batch['segmentation']
        hdmap_labels = batch['hdmap']
        future_egomotion = batch['future_egomotion']
        gt_trajectory = batch['gt_trajectory']

        # present frame hd map gt
        labels['hdmap'] = hdmap_labels[:, self.model.receptive_field - 1].long().contiguous()

        # gt trajectory
        labels['gt_trajectory'] = gt_trajectory

        # Past frames gt depth
        if self.cfg.LIFT.GT_DEPTH:
            depths = batch['depths']
            depth_labels = depths[:, :self.model.receptive_field, :, ::self.model.encoder_downsample,
                           ::self.model.encoder_downsample]
            depth_labels = torch.clamp(depth_labels, self.cfg.LIFT.D_BOUND[0], self.cfg.LIFT.D_BOUND[1] - 1) - \
                           self.cfg.LIFT.D_BOUND[0]
            depth_labels = depth_labels.long().contiguous()
            labels['depths'] = depth_labels

        # Warp labels to present's reference frame
        segmentation_labels_past = cumulative_warp_features(
            segmentation_labels[:, :self.model.receptive_field].float(),
            future_egomotion[:, :self.model.receptive_field],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()[:, :-1]
        segmentation_labels = cumulative_warp_features_reverse(
            segmentation_labels[:, (self.model.receptive_field - 1):].float(),
            future_egomotion[:, (self.model.receptive_field - 1):],
            mode='nearest', spatial_extent=self.spatial_extent,
        ).long().contiguous()
        labels['segmentation'] = torch.cat([segmentation_labels_past, segmentation_labels], dim=1)

        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_labels = batch['pedestrian']
            pedestrian_labels_past = cumulative_warp_features(
                pedestrian_labels[:, :self.model.receptive_field].float(),
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :-1]
            pedestrian_labels = cumulative_warp_features_reverse(
                pedestrian_labels[:, (self.model.receptive_field - 1):].float(),
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()
            labels['pedestrian'] = torch.cat([pedestrian_labels_past, pedestrian_labels], dim=1)

        # Warp instance labels to present's reference frame
        if self.cfg.INSTANCE_SEG.ENABLED:
            gt_instance = batch['instance']
            instance_center_labels = batch['centerness']
            instance_offset_labels = batch['offset']
            gt_instance_past = cumulative_warp_features(
                gt_instance[:, :self.model.receptive_field].float().unsqueeze(2),
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :-1, 0]
            gt_instance = cumulative_warp_features_reverse(
                gt_instance[:, (self.model.receptive_field - 1):].float().unsqueeze(2),
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).long().contiguous()[:, :, 0]
            labels['instance'] = torch.cat([gt_instance_past, gt_instance], dim=1)

            instance_center_labels_past = cumulative_warp_features(
                instance_center_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_center_labels = cumulative_warp_features_reverse(
                instance_center_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['centerness'] = torch.cat([instance_center_labels_past, instance_center_labels], dim=1)

            instance_offset_labels_past = cumulative_warp_features(
                instance_offset_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_offset_labels = cumulative_warp_features_reverse(
                instance_offset_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['offset'] = torch.cat([instance_offset_labels_past, instance_offset_labels], dim=1)

        if self.cfg.INSTANCE_FLOW.ENABLED:
            instance_flow_labels = batch['flow']
            instance_flow_labels_past = cumulative_warp_features(
                instance_flow_labels[:, :self.model.receptive_field],
                future_egomotion[:, :self.model.receptive_field],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()[:, :-1]
            instance_flow_labels = cumulative_warp_features_reverse(
                instance_flow_labels[:, (self.model.receptive_field - 1):],
                future_egomotion[:, (self.model.receptive_field - 1):],
                mode='nearest', spatial_extent=self.spatial_extent,
            ).contiguous()
            labels['flow'] = torch.cat([instance_flow_labels_past, instance_flow_labels], dim=1)

        return labels

    def visualise(self, labels, output, batch_idx, prefix='train'):
        visualisation_video = visualise_output(labels, output, self.cfg)
        name = f'{prefix}_outputs'
        if prefix == 'val':
            name = name + f'_{batch_idx}'
        self.logger.experiment.add_video(name, visualisation_video, global_step=self.training_step_count, fps=2)

    def training_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, True)
        self.training_step_count += 1
        for key, value in loss.items():
            self.logger.experiment.add_scalar('step_train_loss_' + key, value, global_step=self.training_step_count)
        if self.training_step_count % self.cfg.VIS_INTERVAL == 0:
            self.visualise(labels, output, batch_idx, prefix='train')
        return sum(loss.values())

    def validation_step(self, batch, batch_idx):
        output, labels, loss = self.shared_step(batch, False)
        scores = self.metric_vehicle_val.compute()
        self.log('step_val_seg_iou_dynamic', scores[1])
        self.log('step_predicted_traj_x', output['selected_traj'][0, -1, 0])
        self.log('step_target_traj_x', labels['gt_trajectory'][0, -1, 0])
        self.log('step_predicted_traj_y', output['selected_traj'][0, -1, 1])
        self.log('step_target_traj_y', labels['gt_trajectory'][0, -1, 1])

        if batch_idx == 0:
            self.visualise(labels, output, batch_idx, prefix='val')

    def shared_epoch_end(self, step_outputs, is_train):
        if not is_train:
            scores = self.metric_vehicle_val.compute()
            self.logger.experiment.add_scalar('epoch_val_all_seg_iou_dynamic', scores[1],
                                              global_step=self.training_step_count)
            self.metric_vehicle_val.reset()

            if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
                scores = self.metric_pedestrian_val.compute()
                self.logger.experiment.add_scalar('epoch_val_all_seg_iou_pedestrian', scores[1],
                                                  global_step=self.training_step_count)
                self.metric_pedestrian_val.reset()

            if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
                for i, name in enumerate(self.hdmap_class):
                    scores = self.metric_hdmap_val[i].compute()
                    self.logger.experiment.add_scalar('epoch_val_hdmap_iou_' + name, scores[1],
                                                      global_step=self.training_step_count)
                    self.metric_hdmap_val[i].reset()

            if self.cfg.INSTANCE_SEG.ENABLED:
                scores = self.metric_panoptic_val.compute()
                for key, value in scores.items():
                    self.logger.experiment.add_scalar(f'epoch_val_all_ins_{key}_vehicle', value[1].item(),
                                                      global_step=self.training_step_count)
                self.metric_panoptic_val.reset()

            if self.cfg.PLANNING.ENABLED:
                scores = self.metric_planning_val.compute()
                for key, value in scores.items():
                    self.logger.experiment.add_scalar('epoch_val_plan_' + key, value.mean(),
                                                      global_step=self.training_step_count)
                self.metric_planning_val.reset()

        self.logger.experiment.add_scalar('epoch_segmentation_weight',
                                          1 / (2 * torch.exp(self.model.segmentation_weight)),
                                          global_step=self.training_step_count)
        if self.cfg.LIFT.GT_DEPTH:
            self.logger.experiment.add_scalar('epoch_depths_weight', 1 / (2 * torch.exp(self.model.depths_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            self.logger.experiment.add_scalar('epoch_pedestrian_weight',
                                              1 / (2 * torch.exp(self.model.pedestrian_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            self.logger.experiment.add_scalar('epoch_hdmap_weight', 1 / (2 * torch.exp(self.model.hdmap_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.INSTANCE_SEG.ENABLED:
            self.logger.experiment.add_scalar('epoch_centerness_weight',
                                              1 / (2 * torch.exp(self.model.centerness_weight)),
                                              global_step=self.training_step_count)
            self.logger.experiment.add_scalar('epoch_offset_weight', 1 / (2 * torch.exp(self.model.offset_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.INSTANCE_FLOW.ENABLED:
            self.logger.experiment.add_scalar('epoch_flow_weight', 1 / (2 * torch.exp(self.model.flow_weight)),
                                              global_step=self.training_step_count)
        if self.cfg.PLANNING.ENABLED:
            self.logger.experiment.add_scalar('epoch_planning_weight', 1 / (2 * torch.exp(self.model.planning_weight)),
                                              global_step=self.training_step_count)

    def training_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, True)

    def validation_epoch_end(self, step_outputs):
        self.shared_epoch_end(step_outputs, False)

    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(
            params, lr=self.cfg.OPTIMIZER.LR, weight_decay=self.cfg.OPTIMIZER.WEIGHT_DECAY
        )

        return optimizer
