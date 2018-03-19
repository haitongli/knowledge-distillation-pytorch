# knowledge-distillation-pytorch

Exploring knowledge distillation of DNNs for efficient hardware solutions

Author: Haitong Li

Framework: Pytorch


- Usage:
git clone, and install dependencies: pip install -r requirements.txt


- Hyperparameters defined by "params.json" universally within ./experiments/ dir

- Train:

python train.py --model_dir experiments/cnn_distill

python train.py --model_dir experiments/resnet18_distill/resnext_teacher


- Hyperparameter search for an experiment:

python search_hyperparams.py --parent_dir experiments/cnn_distill_alpha_temp

