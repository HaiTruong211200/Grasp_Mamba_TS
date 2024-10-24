# GraspMamba: A Mamba-based Language-driven Grasp Detection Framework with Hierarchical Feature Learning

### Installation
Set up the docker environment, build the docker container for the first time:
```bash
cd GraspMamba/docker
sudo docker build --network host -t graspmamba .
```
Then 
```bash
cd GraspMamba/docker
sudo bash run_container.sh
```
If the code has a problem with triton version, please uninstall triton and install again:
```bash
pip uninstall triton
pip install triton
```

## Datasets
Our dataset can be accessed via [this link](https://airvlab.github.io/grasp-anything/docs/download/). 

### Training model 
Please modify the dataset path based on your setup in the [run_training.sh](/run_training.sh) file. 
```bash
bash run_training.sh
```
### Evaluate model 

```bash
python3 evaluate_test.py --dataset grasp-anything --dataset-path </your dataset path> --seen 1 --model-path </your model path> 
```

## Testing
For testing procedure, by comparing with other grasping methods, please follow the detailed instruction via [LGD](https://github.com/Fsoft-AIC/LGD). 

## Acknowledgement
Our codebase is developed based on [LGD](https://github.com/Fsoft-AIC/LGD) and [MambaVision](https://github.com/NVlabs/MambaVision).