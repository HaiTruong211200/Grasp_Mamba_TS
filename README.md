# GraspMamba: A Mamba-based Language-driven Grasp Detection Framework with Hierarchical Feature Learning

## Lọc và dựng lại dataset từ split

[Quy trình lọc Grasp-Anything++ và lệnh dựng lại từ ZIP](README_DATASET.md).

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

## Grasp-anything Datasets
The dataset can be accessed via [this link](https://airvlab.github.io/grasp-anything/docs/download/). 

### Training model 
Please modify the dataset path based on your setup and other configurations in the [run_training.sh](/run_training.sh) file. Then:
```bash
bash run_training.sh
```
### Evaluate model 

```bash
python3 evaluate_test.py --dataset grasp-anything --dataset-path </your dataset path> --seen <1 or 0>  --model-path </your model path> 
```

## Testing
For testing procedure, by comparing with other grasping methods, please follow the detailed instruction via [LGD](https://github.com/Fsoft-AIC/LGD). 

## Acknowledgement
Our codebase is developed based on [LGD](https://github.com/Fsoft-AIC/LGD) and [MambaVision](https://github.com/NVlabs/MambaVision). Please give us the motivation by citing our work:

```bibtex
@misc{nguyen2024graspmambamambabasedlanguagedrivengrasp,
      title={GraspMamba: A Mamba-based Language-driven Grasp Detection Framework with Hierarchical Feature Learning}, 
      author={Huy Hoang Nguyen and An Vuong and Anh Nguyen and Ian Reid and Minh Nhat Vu},
      year={2024},
      eprint={2409.14403},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.14403}, 
}

```