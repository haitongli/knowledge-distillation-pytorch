# knowledge-distillation-pytorch
* Exploring knowledge distillation of DNNs for efficient hardware solutions
* Author: Haitong Li
* Framework: PyTorch
* Dataset: CIFAR-10


## Features
* A framework for exploring "shallow" and "deep" knowledge distillation (KD) experiments
* Hyperparameters defined by "params.json" universally (avoiding long argparser commands)
* Hyperparameter searching and result synthesizing (as a table)
* Progress bar, tensorboard support, and checkpoint saving/loading (utils.py)
* Pretrained teacher models available for download 


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
* ./model/: teacher and student DNNs, knowledge distillation (KD) loss defination, dataloader 


## Key notes about usage for your experiments:

* Download the zip file for pretrained teacher model checkpoints from this [Box folder](https://stanford.box.com/s/5lwrieh9g1upju0iz9ru93m9d7uo3sox)
* Simply move the unzipped subfolders into 'knowledge-distillation-pytorch/experiments/' (replacing the existing ones if necessary; follow the default path naming)
* Call train.py to start training 5-layer CNN with ResNet-18's dark knowledge, or training ResNet-18 with state-of-the-art deeper models distilled
* Use search_hyperparams.py for hypersearch
* Hyperparameters are defined in params.json files universally. Refer to the header of search_hyperparams.py for details


## Train (dataset: CIFAR-10)

Note: all the hyperparameters can be found and modified in 'params.json' under 'model_dir'

-- Train a 5-layer CNN with knowledge distilled from a pre-trained ResNet-18 model
```
python train.py --model_dir experiments/cnn_distill
```

-- Train a ResNet-18 model with knowledge distilled from a pre-trained ResNext-29 teacher
```
python train.py --model_dir experiments/resnet18_distill/resnext_teacher
```

-- Hyperparameter search for a specified experiment ('parent_dir/params.json')
```
python search_hyperparams.py --parent_dir experiments/cnn_distill_alpha_temp
```

--Synthesize results of the recent hypersearch experiments
```
python synthesize_results.py --parent_dir experiments/cnn_distill_alpha_temp
```


## Results: "Shallow" and "Deep" Distillation

Quick takeaways (more details to be added):

* Knowledge distillation provides regularization for both shallow DNNs and state-of-the-art DNNs
* Having unlabeled or partial dataset can benefit from dark knowledge of teacher models


-**Knowledge distillation from ResNet-18 to 5-layer CNN**

| Model                   | Dropout = 0.5      |  No Dropout        | 
| :------------------:    | :----------------: | :-----------------:|
| 5-layer CNN             | 83.51%             |  84.74%            | 
| 5-layer CNN w/ ResNet18 | 84.49%             |  **85.69%**        |

-**Knowledge distillation from deeper models to ResNet-18**


|Model                      |  Test Accuracy|
|:--------:                 |   :---------: |
|Baseline ResNet-18         | 94.175%       |
|+ KD WideResNet-28-10      | 94.333%       |
|+ KD PreResNet-110         | 94.531%       |
|+ KD DenseNet-100          | 94.729%       |
|+ KD ResNext-29-8          | **94.788%**   |



## References

H. Li, "Exploring knowledge distillation of Deep neural nets for efficient hardware solutions," [CS230 Report](http://cs230.stanford.edu/files_winter_2018/projects/6940224.pdf), 2018

Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y. (2014). Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550.

https://github.com/cs230-stanford/cs230-stanford.github.io

https://github.com/bearpaw/pytorch-classification