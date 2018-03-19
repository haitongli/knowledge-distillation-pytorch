'''Count # of parameters in a trained model'''

import argparse
import os
import numpy as np
import torch
import utils
import model.net as net
import model.resnet as resnet
import model.wrn as wrn
import model.resnext as resnext
import utils


parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory for the dataset")
parser.add_argument('--model', default='resnet18',
                    help="name of the model")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    model_size = 0

    args = parser.parse_args()
    cnn_dir = 'experiments/cnn_distill'
    json_path = os.path.join(cnn_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    if args.model == "resnet18":
        model = resnet.ResNet18()
        model_checkpoint = 'experiments/base_resnet18/best.pth.tar'

    elif args.model == "wrn":
        model = wrn.wrn(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
        model_checkpoint = 'experiments/base_wrn/best.pth.tar'

    elif args.model == "distill_resnext":
        model = resnet.ResNet18()
        model_checkpoint = 'experiments/resnet18_distill/resnext_teacher/best.pth.tar'

    elif args.model == "distill_densenet":
        model = resnet.ResNet18()
        model_checkpoint = 'experiments/resnet18_distill/densenet_teacher/best.pth.tar'

    elif args.model == "cnn":
        model = net.Net(params)
        model_checkpoint = 'experiments/cnn_distill/best.pth.tar'

    utils.load_checkpoint(model_checkpoint, model)

    model_size = count_parameters(model)
    print("Number of parameters in {} is: {}".format(args.model, model_size))