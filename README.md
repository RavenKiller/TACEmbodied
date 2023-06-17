# Embodied experiments for TAC pre-training

## Preparation

### Simulated environments
`run_baseline.py` is for baselines; `run.py` is for baselines+[TAC](https://github.com/RavenKiller/TAC); `VLN-CE/run.py` is for VLN-CE agents.

PointNav, ObjectNav, EQA and Rearrangement require Habitat 2.4; VLN-CE require Habitat 1.7. To setup the environments, please refer to [Habitat-lab](https://github.com/facebookresearch/habitat-lab) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).

### TAC weight
Download the TAC pre-trained depth encoder from [here](https://www.jianguoyun.com/p/DdTCEJwQhY--CRiuxY0FIAA). Put it into `data/checkpoints/TAC/best.pth`.

## Experiments
### PointNav
Train
```
python run.py --config-name=pointnav/ppo_pointnav_gibson.yaml habitat_baselines.evaluate=False
```
Evaluate
```
python run.py --config-name=pointnav/ppo_pointnav_gibson.yaml habitat_baselines.evaluate=True
```
### ObjectNav
Train
```
python run.py --config-name=objectnav/ddppo_objectnav_hm3d.yaml habitat_baselines.evaluate=False
```
Evaluate
```
python run.py --config-name=objectnav/ddppo_objectnav_hm3d.yaml habitat_baselines.evaluate=True
```
### VLN-CE
Train
```
cd VLN-CE
python run.py --run-type train --exp-config vlnce_baselines/config/r2r_baselines/cma_pm.yaml
```
Evaluate
```
cd VLN-CE
python run.py --run-type eval --exp-config vlnce_baselines/config/r2r_baselines/cma_pm.yaml
```
### EQA
Train
```
python run.py --config-name=eqa/il_vqa.yaml habitat_baselines.evaluate=False
python run.py --config-name=eqa/il_pacman_nav.yaml habitat_baselines.evaluate=False
```
Evaluate
```
python run.py --config-name=eqa/il_vqa.yaml habitat_baselines.evaluate=True
python run.py --config-name=eqa/il_pacman_nav.yaml habitat_baselines.evaluate=True
```
### Rearrangement
Train
```
sh train_skills.sh
```
Evaluate
```
python run.py --config-name=rearrange/rl_hierarchical_fixed.yaml habitat_baselines.evaluate=True
```

