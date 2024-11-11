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

* **Performance Evaluation.** Through extensive experiments on the nuScenes dataset, we demonstrate that MomAD significantly outperforms SOTA methods in terms of trajectory consistency and stability, highlighting its effectiveness in tackling challenges within autonomous driving planning. We evaluated the results of long trajectory predictions, specifically at 4, 5, and 6 seconds, which are critical for ensuring the stability of autonomous driving systems.
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
    padding: 2px;">The overall architecture of MomAD. MomAD, as a multi-model trajectory end-to-end autonomous driving method, first encodes multi-view images into feature maps, then learns a sparse scene representation through sparse perception, and finally performs a momentum-guided motion planner to accomplish the planning task. The momentum planning module integrates historical planning to inform current planning, effectively addressing the issue of maximum score deviation in multi-modal trajectories.</div>
</center>


## Results in paper

- Planning results on [nuScenes](https://github.com/nutonomy/nuscenes-devkit).

| Method | Backbone | L2 (m) 1s  | L2 (m) 2s | L2 (m) 3s | L2 (m) Avg | Col. (%) 1s | Col. (%) 2s | Col. (%) 3s | Col. (%) Avg | TPC (m) 1s | TPC (m) 2s | TPC (m) 3s | TPC (m) Avg | FPS |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| UniAD | ResNet101 | 0.45 | 0.70 | 1.04 |0.73 | 0.62 | 0.58 | 0.63 | 0.61 |0.41 | 0.68 | 0.97 | 0.68 |1.8 (A100)|
SparseDrive |ResNet50| **0.29**  | 0.58  | 0.96 |  0.61 |  **0.01** |  **0.05** |  0.18 |  **0.08** |  **0.30** |  0.57 |  0.85 |  0.57 |  **9.0 (4090)**|
**MomAD (Ours)** |  ResNet50    |  0.31 |  **0.57** |  **0.91**  | **0.60** |  **0.01** |  **0.05**  | **0.22**  | 0.09  | **0.30** |  **0.53** |  **0.78** |  **0.54** |  7.8 (4090) | 

- Planning results for long trajectory prediction on [nuScenes](https://github.com/nutonomy/nuscenes-devkit).  We train 10 epochs on 6s trajectories and test on 6s trajectories.

| Method | L2 (m) 4s  | L2 (m) 5s | L2 (m) 6s | Col. (%) 4s | Col. (%) 5s | Col. (%) 6s | TPC (m) 4s | TPC (m) 5s | TPC (m) 6s |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
SparseDrive   | 1.75|2.32|2.95| 0.87| 1.54| 2.33| 1.33| 1.66| 1.99|
**MomAD (Ours)** |  **1.67**| **1.98**| **2.45**| **0.83**| **1.43**| **2.13**| **1.19**| **1.45**| **1.61**|

- Planning results on the Turning-nuScenes validation dataset [Turning-nuScenes ](nuscenes_infos_val_hrad_planing_scene.pkl).  We train 10 epochs on 6s trajectories and test on 6s trajectories.

| Method | L2 (m) 4s  | L2 (m) 5s | L2 (m) 6s | Col. (%) 4s | Col. (%) 5s | Col. (%) 6s | TPC (m) 4s | TPC (m) 5s | TPC (m) 6s |
| :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
SparseDrive   | 1.75|2.32|2.95| 0.87| 1.54| 2.33| 1.33| 1.66| 1.99|
**MomAD (Ours)** |  **1.67**| **1.98**| **2.45**| **0.83**| **1.43**| **2.13**| **1.19**| **1.45**| **1.61**|

## Trajectory Prediction Consistency (TPC) metric
To evaluate the planning stability of MomAD, we propose a new [Trajectory Prediction Consistency (TPC) metric](/projects/mmdet3d_plugin/datasets/evaluation/planning/planning_eval_roboAD_6s.py) to measure consistency between predicted and historical trajectories.

## 


## Quick Start
[Quick Start](docs/quick_start.md)

## Acknowledgement
- [SparseDrive](https://github.com/swc-17/SparseDrive)
- [UniAD](https://github.com/OpenDriveLab/UniAD) 
- [VAD](https://github.com/hustvl/VAD)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

