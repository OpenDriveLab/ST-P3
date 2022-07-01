# ST-P3

The core idea is to devise an **interpretable** pipeline to generate planning trajectories directly from image-only inputs. The codebase is inherited from [FIERY](https://github.com/wayveai/fiery).

TODO(demo video)

> **ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning**  
> Shengchao Hu, Li Chen, Penghao Wu, [Hongyang Li](https://lihongyang.info/), [Junchi Yan](https://thinklab.sjtu.edu.cn/), Dacheng Tao.       
> [paper](https://arxiv.org/abs/) | [blog(Chinese)]()

## 🚗 Introduction

TODO(abstract)

TODO(fig)

## ⚙ Setup
- Create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment by running `conda env create -f environment.yml`.

## 🏄 Evaluation
- Download the [nuScenes dataset](https://www.nuscenes.org/download).
- Download the pretrained weights.
- Run `python evaluate.py --checkpoint ${CHECKPOINT_PATH}`.

## 🔥 Pre-trained models
- Perceive
- Prediction
- Planning

## 🏊 Training
To train the model from scratch on NuScenes:
- Run `python train.py --config fiery/configs/nuscenes/baseline.yml`.

To train on single GPU add the flag `GPUS [0]`, and to change the batch size use the flag `BATCHSIZE ${DESIRED_BATCHSIZE}`.

## ✏️ Citation

If you find our repo or our paper useful, please use the following citation:

```
@article{hu2022stp3,
 title={ST-P3: End-to-end Vision-based Autonomous Driving via Spatial-Temporal Feature Learning}, 
 author={Shengchao Hu and Li Chen and Penghao Wu and Hongyang Li and Junchi Yan and Dacheng Tao},
 journal={arXiv preprint arXiv:},
 year={2022},
}
```

## ©️ License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## 🙌 Acknowledgement
We thank Xiangwei Geng for his support on the depth map generation, and fruitful discussions from [Xiaosong Jia](https://jiaxiaosong1002.github.io/). We have many thanks to [FIERY](https://github.com/wayveai/fiery) team for their exellent open source project.
