"""Analyzes, visualizes knowledge distillation"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import utils
import model.net as net
import model.resnet as resnet
import model.data_loader as data_loader
from torchnet.meter import ConfusionMeter
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory of params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--dataset', default='dev', help="dataset to analze the model on")
parser.add_argument('--temperature', type=float, default=1.0, \
                    help="temperature used for softmax output")


def model_analysis(model, dataloader, params, temperature=1., num_classes=10):
    """
        Generate Confusion Matrix on evaluation set
    """
    model.eval()
    confusion_matrix = ConfusionMeter(num_classes)
    softmax_scores = []
    predict_correct = []

    with tqdm(total=len(dataloader)) as t:
        for idx, (data_batch, labels_batch) in enumerate(dataloader):

            if params.cuda:
                data_batch, labels_batch = data_batch.cuda(async=True), \
                                           labels_batch.cuda(async=True)
            data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)

            output_batch = model(data_batch)

            confusion_matrix.add(output_batch.data, labels_batch.data)

            softmax_scores_batch = F.softmax(output_batch/temperature, dim=1)
            softmax_scores_batch = softmax_scores_batch.data.cpu().numpy()
            softmax_scores.append(softmax_scores_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            predict_correct_batch = (np.argmax(output_batch, axis=1) == labels_batch).astype(int)
            predict_correct.append(np.reshape(predict_correct_batch, (labels_batch.size, 1)))

            t.update()

    softmax_scores = np.vstack(softmax_scores)
    predict_correct = np.vstack(predict_correct)

    return softmax_scores, predict_correct, confusion_matrix.value().astype(int)


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'analysis.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # fetch dataloaders
    # train_dl = data_loader.fetch_dataloader('train', params)
    # dev_dl = data_loader.fetch_dataloader('dev', params)
    dataloader = data_loader.fetch_dataloader(args.dataset, params)

    logging.info("- done.")

    # Define the model graph
    model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18()

    # fetch loss function and metrics
    metrics = resnet.metrics
    
    logging.info("Starting analysis...")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate and analyze
    softmax_scores, predict_correct, confusion_matrix = model_analysis(model, dataloader, params,
                                                                       args.temperature)

    results = {'softmax_scores': softmax_scores, 'predict_correct': predict_correct,
               'confusion_matrix': confusion_matrix}

    for k, v in results.items():
        filename = args.dataset + '_temp' + str(args.temperature) + '_' + k + '.txt'
        save_path = os.path.join(args.model_dir, filename)
        np.savetxt(save_path, v)