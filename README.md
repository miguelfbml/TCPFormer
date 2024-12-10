
# AAAI 2025
# TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation


---

## Environment
The project is developed under the following environment:
- Python 3.10.x
- PyTorch 2.2.1
- CUDA 12.1
For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 

## Dataset
### Human3.6M
#### Preprocessing
1. We follow the previous state-of-the-art method [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md) for dataset setup. Download the [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:

**For our model with T = 243**:
```text
python h36m.py  --n-frames 243
```
**or T = 81**
```text
python h36m.py  --n-frames 81
```
**or T = 27**
```text
python h36m.py  --n-frames 81
```


### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

## Training
After dataset preparation, you can train the model as follows:
### Human3.6M
You can train Human3.6M with the following command:
```
python train.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/h36m`. 
```
python train.py --config configs/h36m/TCPFormer_h36m_243.yaml 
```
### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python train_3dhp.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/mpi`. 
```
python train_3dhp.py --config configs/mpi/TCPFormer_mpi_81.yaml 
```


## Evaluation
| Dataset  | frames | Checkpoint|
|----------|--------|-----------|
|Human3.6M|81|[download](https://drive.google.com/file/d/14D_gfCflgl67-nl0L2MJijbARizbphnP/view?usp=drive_link)|
|Human3.6M|243|[download](https://drive.google.com/file/d/1xiCQaYOWlNBR4uZVGmFJ644mB4tPH-Gq/view?usp=drive_link)|
|MPI-INF-3DHP|9|[download](https://drive.google.com/file/d/1z_foxtKFxz1_g8jOfP-_cqv7ciptpJNo/view?usp=drive_link)|
|MPI-INF-3DHP|27|[download](https://drive.google.com/file/d/1EHl7IFud3JkDmDsDK6vad7O4STAMp9T_/view?usp=drive_link)|
|MPI-INF-3DHP|81|[download](https://drive.google.com/file/d/1ST3NYm-xlgkrMhs3nHm6_WVt6jvCzL-e/view?usp=drive_link)|




After downloading the weight from table above, you can evaluate Human3.6M models by:
```
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
For example if TCPFormer with T = 243 of H.36M is downloaded and put in `checkpoint` directory, then you can run:
```
python train.py --eval-only  --checkpoint checkpoint --checkpoint-file TCPFormer_h36m_243_379.pth.tr --config configs/h36m/TCPFormer_h36m_243.yaml
```

Similarly, TCPFormer with T = 81 of H.36M is downloaded and put in `checkpoint` directory, then you can run:
```
python train.py --eval-only  --checkpoint checkpoint --checkpoint-file TCPFormer_h36m_81_405.pth.tr --config configs/h36m/TCPFormer_h36m_81.yaml
```



For MPI-INF-3DHP dataset, you can download the checkpoint with T = 81 and put in `checkpoint_mpi` directory, then you can run:
```
python train_3dhp.py --eval-only  --checkpoint checkpoint_mpi --checkpoint-file TCPFormer_mpi_81.pth.tr --config configs/mpi/TCPFormer_mpi_81.yaml
```
