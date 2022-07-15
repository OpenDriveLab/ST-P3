import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stp3.utils.tools import gen_dx_bx
from stp3.utils.geometry import calculate_birds_eye_view_parameters
from skimage.draw import polygon


class Cost_Function(nn.Module):
    def __init__(self, cfg):
        super(Cost_Function, self).__init__()

        self.safetycost = SafetyCost(cfg)
        self.headwaycost = HeadwayCost(cfg)
        self.lrdividercost = LR_divider(cfg)
        self.comfortcost = Comfort(cfg)
        self.progresscost = Progress(cfg)
        self.rulecost = Rule(cfg)
        self.costvolume = Cost_Volume(cfg)

        self.n_future = cfg.N_FUTURE_FRAMES


    def forward(self, cost_volume, trajs, semantic_pred, lane_divider, drivable_area, target_point):
        '''
        cost_volume: torch.Tensor<float> (B, n_future, 200, 200)
        trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
        semantic_pred: torch.Tensor<float> (B, n_future, 200, 200)
        drivable_area: torch.Tensor<float> (B, 1/2, 200, 200)
        lane_divider: torch.Tensor<float> (B, 1/2, 200, 200)
        target_points: torch.Tensor<float> (B, 2)
        '''
        trajs = trajs * torch.tensor([-1, 1], device=trajs.device)
        safetycost = torch.clamp(self.safetycost(trajs, semantic_pred), 0, 100)
        headwaycost = torch.clamp(self.headwaycost(trajs, semantic_pred, drivable_area), 0, 100)
        lrdividercost = torch.clamp(self.lrdividercost(trajs, lane_divider), 0, 100)
        comfortcost = torch.clamp(self.comfortcost(trajs), 0, 100)
        progresscost = torch.clamp(self.progresscost(trajs, target_point), -100, 100)
        rulecost = torch.clamp(self.rulecost(trajs, drivable_area), 0, 100)
        costvolume = torch.clamp(self.costvolume(trajs, cost_volume), 0, 100)

        cost_fo = safetycost + headwaycost + lrdividercost + costvolume + rulecost
        cost_fc = comfortcost + progresscost

        return cost_fc, cost_fo



class BaseCost(nn.Module):
    def __init__(self, cfg):
        super(BaseCost, self).__init__()
        self.cfg = cfg

        dx, bx, _ = gen_dx_bx(self.cfg.LIFT.X_BOUND, self.cfg.LIFT.Y_BOUND, self.cfg.LIFT.Z_BOUND)
        dx, bx = dx[:2], bx[:2]
        self.dx = nn.Parameter(dx,requires_grad=False)
        self.bx = nn.Parameter(bx,requires_grad=False)

        _,_, self.bev_dimension = calculate_birds_eye_view_parameters(
            cfg.LIFT.X_BOUND, cfg.LIFT.Y_BOUND, cfg.LIFT.Z_BOUND
        )

        self.W = cfg.EGO.WIDTH
        self.H = cfg.EGO.HEIGHT


    def get_origin_points(self, lambda_=0):
        W = self.W
        H = self.H
        pts = np.array([
            [-H / 2. + 0.5 - lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, W / 2. + lambda_],
            [H / 2. + 0.5 + lambda_, -W / 2. - lambda_],
            [-H / 2. + 0.5 - lambda_, -W / 2. - lambda_],
        ])
        pts = (pts - self.bx.cpu().numpy()) / (self.dx.cpu().numpy())
        pts[:, [0, 1]] = pts[:, [1, 0]]
        rr , cc = polygon(pts[:,1], pts[:,0])
        rc = np.concatenate([rr[:,None], cc[:,None]], axis=-1)
        return torch.from_numpy(rc).to(device=self.bx.device) # (27,2)

    def get_points(self, trajs, lambda_=0):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)
        return:
        List[ torch.Tensor<int> (B, N, n_future), torch.Tensor<int> (B, N, n_future)]
        '''
        rc = self.get_origin_points(lambda_)
        B, N, n_future, _ = trajs.shape

        trajs = trajs.view(B, N, n_future, 1, 2) / self.dx
        trajs[:,:,:,:,[0,1]] = trajs[:,:,:,:,[1,0]]
        trajs = trajs + rc

        rr = trajs[:,:,:,:,0].long()
        rr = torch.clamp(rr, 0, self.bev_dimension[0] - 1)

        cc = trajs[:,:,:,:,1].long()
        cc = torch.clamp(cc, 0, self.bev_dimension[1] - 1)

        return rr, cc

    def compute_area(self, semantic_pred, trajs, ego_velocity=None, _lambda=0):
        '''
        semantic: torch.Tensor<float> (B, n_future, 200, 200)
        trajs: torch.Tensor<float> (B, N, n_future, 2)
        ego_velocity: torch.Tensor<float> (B, N, n_future)
        '''
        _lambda = int(_lambda / self.dx[0])
        rr, cc = self.get_points(trajs, _lambda)
        B, N, n_future,_ = trajs.shape

        if ego_velocity is None:
            ego_velocity = torch.ones((B,N,n_future), device=semantic_pred.device)

        ii = torch.arange(B)
        kk = torch.arange(n_future)
        subcost = semantic_pred[ii[:, None, None, None], kk[None, None, :, None], rr, cc].sum(dim=-1)
        subcost = subcost * ego_velocity


        return subcost

    def discretize(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
        '''
        B, N, n_future, _ = trajs.shape

        xx, yy = trajs[:,:,:,0], trajs[:,:,:,1]

        # discretize
        yi = ((yy - self.bx[0]) / self.dx[0]).long()
        yi = torch.clamp(yi,0, self.bev_dimension[0]-1)

        xi = ((xx - self.bx[1]) / self.dx[1]).long()
        xi = torch.clamp(xi, 0, self.bev_dimension[1]-1)

        return yi, xi

    def evaluate(self, trajs, C):
        '''
            trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
            C: torch.Tensor<float> (B, n_future, 200, 200)
        '''
        B, N, n_future, _ = trajs.shape

        ii = torch.arange(B)
        ti = torch.arange(n_future)

        Syi, Sxi = self.discretize(trajs)

        CS = C[ii[:, None, None], ti[None, None, :], Syi, Sxi]
        return CS

class Cost_Volume(BaseCost):
    def __init__(self, cfg):
        super(Cost_Volume, self).__init__(cfg)

        self.factor = cfg.COST_FUNCTION.VOLUME

    def forward(self, trajs, cost_volume):
        '''
        cost_volume: torch.Tensor<float> (B, n_future, 200, 200)
        trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
        '''

        cost_volume = torch.clamp(cost_volume, 0, 1000)

        return self.evaluate(trajs, cost_volume) * self.factor

class Rule(BaseCost):
    def __init__(self, cfg):
        super(Rule, self).__init__(cfg)

        self.factor = 5

    def forward(self, trajs, drivable_area):
        '''
            trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
            drivable_area: torch.Tensor<float> (B, 1/2, 200, 200)
        '''
        assert drivable_area.ndim == 4, 'drivable area ndim should be 4'
        if drivable_area.shape[1] == 2:
            drivable_area = torch.softmax(drivable_area, dim=1)[:, 1]
            mask = drivable_area < 0.5
            drivable_area[mask] = 0
        else:
            drivable_area = drivable_area[:, 0]

        B, _, n_future, _ = trajs.shape
        _,H,W = drivable_area.shape

        dangerous_area = torch.logical_not(drivable_area).float().view(B, 1, W, H).expand(B, n_future, W, H)
        subcost = self.compute_area(dangerous_area, trajs)

        return subcost * self.factor


class SafetyCost(BaseCost):
    def __init__(self, cfg):
        super(SafetyCost, self).__init__(cfg)
        self.w = nn.Parameter(torch.tensor([1.,1.]),requires_grad=False)

        self._lambda = cfg.COST_FUNCTION.LAMBDA
        self.factor = cfg.COST_FUNCTION.SAFETY

    def forward(self, trajs, semantic_pred):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
        semantic_pred: torch.Tensor<float> (B, n_future, 200, 200)
        ego_velocity: torch.Tensor<float> (B, N, n_future)
        '''
        B, N, n_future, _ = trajs.shape
        ego_velocity = torch.zeros((B,N,n_future), device=semantic_pred.device)
        for i in range(n_future):
            if i == 0:
                ego_velocity[:,:,i] = torch.sqrt((trajs[:,:,i] ** 2).sum(axis=-1)) / 0.5
            else:
                ego_velocity[:,:,i] = torch.sqrt(((trajs[:,:,i] - trajs[:,:,i-1]) ** 2).sum(dim=-1)) / 0.5

        # o_c(tau, t, 0)
        subcost1 = self.compute_area(semantic_pred, trajs)
        # o_c(tau, t, lambda) x v(tau, t)
        subcost2 = self.compute_area(semantic_pred, trajs, ego_velocity, self._lambda)

        subcost = subcost1 * self.w[0] + subcost2 * self.w[1]


        return subcost * self.factor


class HeadwayCost(BaseCost):
    def __init__(self, cfg):
        super(HeadwayCost, self).__init__(cfg)
        self.L = 10  # Longitudinal distance keep 10m
        self.factor = cfg.COST_FUNCTION.HEADWAY

    def forward(self, trajs, semantic_pred, drivable_area):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
        semantic_pred: torch.Tensor<float> (B, n_future, 200, 200)
        drivable_area: torch.Tensor<float> (B, 1/2, 200, 200)
        '''
        assert drivable_area.ndim == 4, 'drivable area ndim should be 4'
        if drivable_area.shape[1] == 2:
            drivable_area = torch.softmax(drivable_area, dim=1)[:, 1]
            mask = drivable_area < 0.5
            drivable_area[mask] = 0
        else:
            drivable_area = drivable_area[:, 0]
        B, N, n_future, _ = trajs.shape
        _, W, H = drivable_area.shape
        drivable_mask = drivable_area.view(B, 1, W, H).expand(B, n_future, W, H)
        semantic_pred_ = semantic_pred * drivable_mask
        tmp_trajs = trajs.clone()
        tmp_trajs[:,:,:,1] = tmp_trajs[:,:,:,1]+self.L

        subcost = self.compute_area(semantic_pred_, tmp_trajs)

        return subcost * self.factor

class LR_divider(BaseCost):
    def __init__(self, cfg):
        super(LR_divider, self).__init__(cfg)
        self.L = 1 # Keep a distance of 2m from the lane line
        self.factor = cfg.COST_FUNCTION.LRDIVIDER

    def forward(self, trajs, lane_divider):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)   N: sample number
        lane_divider: torch.Tensor<float> (B, 1/2, 200, 200)
        '''
        assert lane_divider.ndim == 4, 'lane_divider ndim should be 4'
        if lane_divider.shape[1] == 2:
            lane_divider = torch.softmax(lane_divider, dim=1)[:,1]
            mask = lane_divider <= 0.5
            lane_divider[mask] = 0
        else:
            lane_divider = lane_divider[:,0]

        B, N, n_future, _ = trajs.shape

        yy, xx = self.discretize(trajs)
        yx = torch.stack([yy,xx],dim=-1) # (B, N, n_future, 2)

        # lane divider
        res1 = []
        for i in range(B):
            index = torch.nonzero(lane_divider[i]) # (n, 2)
            if len(index) != 0:
                yx_batch = yx[i].view(N, n_future, 1, 2)
                distance = torch.sqrt((((yx_batch - index) * reversed(self.dx))**2).sum(dim=-1)) # (N, n_future, n)
                distance,_ = distance.min(dim=-1) # (N, n_future)
                index = distance > self.L
                distance = (self.L - distance) ** 2
                distance[index] = 0
            else:
                distance = torch.zeros((N, n_future),device=trajs.device)
            res1.append(distance)
        res1 = torch.stack(res1, dim=0)

        return res1 * self.factor


class Comfort(BaseCost):
    def __init__(self, cfg):
        super(Comfort, self).__init__(cfg)

        self.c_lat_acc = 3 # m/s2
        self.c_lon_acc = 3 # m/s2
        self.c_jerk = 1 # m/s3

        self.factor = cfg.COST_FUNCTION.COMFORT

    def forward(self, trajs):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)
        '''
        B, N, n_future, _ = trajs.shape
        lateral_velocity = torch.zeros((B,N,n_future), device=trajs.device)
        longitudinal_velocity = torch.zeros((B, N, n_future), device=trajs.device)
        lateral_acc = torch.zeros((B,N,n_future), device=trajs.device)
        longitudinal_acc = torch.zeros((B,N,n_future), device=trajs.device)
        for i in range(n_future):
            if i == 0:
                lateral_velocity[:,:,i] = trajs[:,:,i,0] / 0.5
                longitudinal_velocity[:,:,i] = trajs[:,:,i,1] / 0.5
            else:
                lateral_velocity[:,:,i] = (trajs[:,:,i,0] - trajs[:,:,i-1,0]) / 0.5
                longitudinal_velocity[:,:,i] = (trajs[:,:,i,1] - trajs[:,:,i-1,1]) / 0.5
        for i in range(1, n_future):
            lateral_acc[:,:,i] = (lateral_velocity[:,:,i] - lateral_velocity[:,:,i-1]) / 0.5
            longitudinal_acc[:,:,i] = (longitudinal_velocity[:,:,i] - longitudinal_velocity[:,:,i-1]) / 0.5
        lateral_acc, _ = torch.abs(lateral_acc).max(dim=-1)
        longitudinal_acc, _ = torch.abs(longitudinal_acc).max(dim=-1)
        # v^2 - v_0^2 = 2ax
        # lateral_acc = (lateral_velocity[:,:,-1] ** 2 - lateral_velocity[:,:,0] ** 2) / (2 * (trajs[:,:,-1,0] - trajs[:,:,0,0]))
        # longitudinal_acc = (longitudinal_velocity[:,:,-1] ** 2 - longitudinal_velocity[:,:,0] ** 2) / (2 * (trajs[:,:,-1,1] - trajs[:,:,0,1]))
        # index = torch.isnan(lateral_acc)
        # lateral_acc[index] = 0.0
        # index = torch.isnan(longitudinal_acc)
        # longitudinal_acc[index] = 0.0

        # jerk
        ego_velocity = torch.zeros((B, N, n_future), device=trajs.device)
        ego_acc = torch.zeros((B,N,n_future), device=trajs.device)
        ego_jerk = torch.zeros((B,N,n_future), device=trajs.device)
        for i in range(n_future):
            if i == 0:
                ego_velocity[:, :, i] = torch.sqrt((trajs[:, :, i] ** 2).sum(dim=-1)) / 0.5
            else:
                ego_velocity[:, :, i] = torch.sqrt(((trajs[:, :, i] - trajs[:, :, i - 1]) ** 2).sum(dim=-1)) / 0.5
        for i in range(1, n_future):
            ego_acc[:,:,i] = (ego_velocity[:,:,i] - ego_velocity[:,:,i-1]) / 0.5
        for i in range(2, n_future):
            ego_jerk[:,:,i] = (ego_acc[:,:,i] - ego_acc[:,:,i-1]) / 0.5
        ego_jerk,_ = torch.abs(ego_jerk).max(dim=-1)

        subcost = torch.zeros((B, N), device=trajs.device)

        lateral_acc = torch.clamp(torch.abs(lateral_acc) - self.c_lat_acc, 0,30)
        subcost += lateral_acc ** 2
        longitudinal_acc = torch.clamp(torch.abs(longitudinal_acc) - self.c_lon_acc, 0, 30)
        subcost += longitudinal_acc ** 2
        ego_jerk = torch.clamp(torch.abs(ego_jerk) - self.c_jerk, 0, 20)
        subcost += ego_jerk ** 2

        return subcost * self.factor

class Progress(BaseCost):
    def __init__(self, cfg):
        super(Progress, self).__init__(cfg)
        self.factor = cfg.COST_FUNCTION.PROGRESS

    def forward(self, trajs, target_points):
        '''
        trajs: torch.Tensor<float> (B, N, n_future, 2)
        target_points: torch.Tensor<float> (B, 2)
        '''
        B, N, n_future, _ = trajs.shape
        subcost1, _ = trajs[:,:,:,1].max(dim=-1)

        if target_points.sum() < 0.5:
            subcost2 = 0
        else:
            trajs = trajs[:,:,-1]
            target_points = target_points.unsqueeze(1)
            subcost2 = ((trajs - target_points) ** 2).sum(dim=-1)

        return (subcost2 - subcost1) * self.factor
