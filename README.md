# Don’t Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving

## Abstract
<div align="justify">  
End-to-end autonomous driving frameworks facilitate seamless integration of perception and planning but often rely on one-shot trajectory prediction, lacking temporal consistency and long-horizon awareness. This limitation can lead to unstable control, undesirable shifts, and vulnerability to occlusions in single-frame perception. In this work, we propose the Momentum-Aware Driving (MomAD) framework to address these issues by introducing trajectory momentum and perception momentum to stabilize and refine trajectory prediction. MomAD consists of two key components: (1) Topological Trajectory Matching (TTM), which uses Hausdorff Distance to align predictions with prior paths and ensure temporal coherence, and (2) Momentum Planning Interactor (MPI), which cross-attends the planning query with historical spatial-temporal context. Additionally, an encoder-decoder module introduces feature perturbations to increase robustness against perception noise. To quantify planning stability, we propose the Trajectory Prediction Consistency (TPC) metric, showing that MomAD achieves long-term consistency (>3s) on the nuScenes dataset. We further curate the challenging Turning-nuScenes validation set, focused on turning scenarios, where MomAD surpasses state-of-the-art methods, highlighting its enhanced stability and responsiveness in dynamic driving conditions.
</div>
<div align="justify">  
:fire: Contributions:
* **Momentum Planning Concept.** We propose the concept of momentum planning in multi-modal trajectory planning, drawing an analogy to human driving behavior. We provide theoretical evidence to demonstrate the effectiveness of our momentum planning in addressing temporal consistency in end-to-end autonomous driving

* **MomAD Framework.** We propose MomAD, an end-to-end autonomous driving framework that employs momentum planning. It optimizes current trajectory planning by integrating historical planning guidance, significantly improving trajectory consistency and stability in autonomous driving.

* **Turning NuScenes Validation Dataset.** We create the Turning-nuScenes val dataset, derived from the nuScenes full validation dataset. This new dataset focuses on turning scenarios, providing a specialized benchmark for evaluating the performance of autonomous driving systems in complex driving situations.

* **Trajectory Prediction Consistency (TPC) Metric.** We introduce the TPC metric to quantitatively assess the consistency of trajectory predictions in existing end-to-end autonomous driving methods, addressing a critical gap in the evaluation of trajectory planning.

***Performance Evaluation.** Through extensive experiments on the nuScenes dataset, we demonstrate that MomAD significantly outperforms SOTA methods in terms of trajectory consistency and stability, highlighting its effectiveness in tackling challenges within autonomous driving planning. We evaluated the results of long trajectory predictions, specifically at 4, 5, and 6 seconds, which are critical for ensuring the stability of autonomous driving systems.
</div>
## Method
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="main.png" width="1000">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">The overall architecture of **MomAD**. **MomAD**, as a multi-model trajectory end-to-end autonomous driving method, first encodes multi-view images into feature maps, then learns a sparse scene representation through sparse perception, and finally performs a momentum-guided motion planner to accomplish the planning task. The momentum planning module integrates historical planning to inform current planning, effectively addressing the issue of maximum score deviation in multi-modal trajectories.</div>
</center>


## Results in paper

- Comprehensive results for all tasks on [nuScenes](https://github.com/nutonomy/nuscenes-devkit).

| Method | NDS | AMOTA | minADE (m) | L2 (m) Avg | Col. (%) Avg | Training Time (h) | FPS |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.498 | 0.359 | 0.71 | 0.73 | 0.61 | 144 | 1.8 |
| SparseDrive-S | 0.525 | 0.386 | 0.62 | 0.61 | 0.08 | **20** | **9.0** |
| SparseDrive-B | **0.588** | **0.501** | **0.60** | **0.58** | **0.06** | 30 | 7.3 |

- Open-loop planning results on [nuScenes](https://github.com/nutonomy/nuscenes-devkit).

| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg | FPS |
| :---: | :---: | :---: | :---: | :---:| :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.45 | 0.70 | 1.04 | 0.73 | 0.62 | 0.58 | 0.63 | 0.61 | 1.8 |
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.03 | 0.19 | 0.43 | 0.21 |4.5 |
| SparseDrive-S | **0.29** | 0.58 | 0.96 | 0.61 | 0.01 | 0.05 | 0.18 | 0.08 | **9.0** |
| SparseDrive-B | **0.29** | **0.55** | **0.91** | **0.58** | **0.01** | **0.02** | **0.13** | **0.06** | 7.3 |

## Results of released checkpoint
We found that some collision cases were not taken into consideration in our previous code, so we re-implement the evaluation metric for collision rate in released code and provide updated results.

## Main results
| Model | config | ckpt | log | det: NDS | mapping: mAP | track: AMOTA |track: AMOTP | motion: EPA_car |motion: minADE_car| motion: minFDE_car | motion: MissRate_car | planning: CR | planning: L2 |
| :---: | :---: | :---: | :---: | :---: | :---:|:---:|:---: | :---: | :----: | :----: | :----: | :----: | :----: |
| Stage1 |[cfg](projects/configs/sparsedrive_small_stage1.py)|[ckpt](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1.pth)|[log](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage1_log.txt)|0.5260|0.5689|0.385|1.260| | | | | | |
| Stage2 |[cfg](projects/configs/sparsedrive_small_stage2.py)|[ckpt](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth)|[log](https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2_log.txt)|0.5257|0.5656|0.372|1.248|0.492|0.61|0.95|0.133|0.097%|0.61|

## Detailed results for planning
| Method | L2 (m) 1s | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD | 0.45 | 0.70 | 1.04 | 0.73 | 0.66 | 0.66 | 0.72 | 0.68 |
| UniAD-wo-post-optim | 0.32 | 0.58 | 0.94 | 0.61 | 0.17 | 0.27 | 0.42 | 0.29 |
| VAD | 0.41 | 0.70 | 1.05 | 0.72 | 0.03 | 0.21 | 0.49 | 0.24 | 
| SparseDrive-S | 0.30 | 0.58 | 0.95 | 0.61 | 0.01 | 0.05 | 0.23 | 0.10 | 


## Quick Start
[Quick Start](docs/quick_start.md)

## Citation
If you find SparseDrive useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.
```
@article{sun2024sparsedrive,
  title={SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation},
  author={Sun, Wenchao and Lin, Xuewu and Shi, Yining and Zhang, Chuang and Wu, Haoran and Zheng, Sifa},
  journal={arXiv preprint arXiv:2405.19620},
  year={2024}
}
```

## Acknowledgement
- [Sparse4D](https://github.com/HorizonRobotics/Sparse4D)
- [UniAD](https://github.com/OpenDriveLab/UniAD) 
- [VAD](https://github.com/hustvl/VAD)
- [StreamPETR](https://github.com/exiawsh/StreamPETR)
- [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

