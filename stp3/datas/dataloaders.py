import torch
import torch.utils.data
from nuscenes.nuscenes import NuScenes
from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.datas.CarlaData import CarlaDataset


def prepare_dataloaders(cfg, return_dataset=False):
    if cfg.DATASET.NAME == 'nuscenes':
        # 28130 train and 6019 val
        dataroot = cfg.DATASET.DATAROOT
        nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
        traindata = FuturePredictionDataset(nusc, 0, cfg)
        valdata = FuturePredictionDataset(nusc, 1, cfg)

        if cfg.DATASET.VERSION == 'mini':
            traindata.indices = traindata.indices[:10]
            # valdata.indices = valdata.indices[:10]

        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    elif cfg.DATASET.NAME == 'carla':
        dataroot = cfg.DATASET.DATAROOT
        traindata = CarlaDataset(dataroot, True, cfg)
        valdata = CarlaDataset(dataroot, False, cfg)
        nworkers = cfg.N_WORKERS
        trainloader = torch.utils.data.DataLoader(
            traindata, batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=nworkers, pin_memory=True, drop_last=True
        )
        valloader = torch.utils.data.DataLoader(
            valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False)
    else:
        raise NotImplementedError

    if return_dataset:
        return trainloader, valloader, traindata, valdata
    else:
        return trainloader, valloader