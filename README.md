# knowledge-distillation-pytorch
* Exploring knowledge distillation of DNNs for efficient hardware solutions
* Author: Haitong Li
* Tool: Pytorch


## Features
* A framework for exploring knowledge distillation (KD) experiments with shallow/deep models
* Hyperparameters defined by "params.json" universally (avoiding long argparser commands)
* Easy hyperparameter searching and result synthesizing (as a table)
* Progress bar, tensorboard support, and checkpoint saving/loading (utils.py)


## Install
* Clone the repo
  ```
  git clone https://github.com/peterliht/knowledge-distillation-pytorch.git
  ```

* Install the dependencies (including Pytorch)
  ```
  pip install -r requirements.txt
  ```


## Organizatoin:
* ./train.py: main entrance for train/eval with or without KD on CIFAR-10
* ./experiments/: json files for each experiment; dir for hypersearch
* ./model/: pre-defined teacher/student DNNs, dataloader, plus knowledge distillation (KD) loss
* ./data_analysis/: matplotlib scripts for data analysis on KD 


## Note:

* Since training with KD requires access to pre-trained models (in the eval() state), I will upload "best.pth.tar" checkpoints for teacher models (WideResNet, ResNext, PreResNet, DenseNet) somewhere else soon

* Meanwhile, ResNet18 can be trained using train.py with specificed model name, which can then be used as the teacher model to train shallow CNN as described below

## Train (dataset: CIFAR-10)

Note: all the hyperparameters can be found and modified in 'params.json' files

-- Train a 5-layer CNN with knowledge distilled from a pre-trained ResNet18 model
```
python train.py --model_dir experiments/cnn_distill
```

-- Train a ResNet18 model with knowledge distilled from a pre-trained ResNext29 teacher
```
python train.py --model_dir experiments/resnet18_distill/resnext_teacher
```

-- Hyperparameter search for a specified experiment ('parent_dir/params.json')
```
python search_hyperparams.py --parent_dir experiments/cnn_distill_alpha_temp
```

--Synthesize results of the hypersearch
```
python synthesize_results.py --parent_dir experiments/cnn_distill_alpha_temp
```


## Results
Quick takeaway (more details to be added):

* Knowledge distillation provides regularization for both shallow DNNs and state-of-the-art DNNs
* KD can also help in the scenarios of using unlabeled dataset and small amount of data for training

## References
https://github.com/cs230-stanford/cs230-stanford.github.io

https://github.com/bearpaw/pytorch-classification

https://github.com/kuangliu/pytorch-cifar