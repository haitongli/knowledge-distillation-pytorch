"""
   Peform hyperparemeters search

   A brief definition/clarification of 'params.json' files:

   "model_version": "resnet18", # "base" models or "modelname"_distill models
   "subset_percent": 1.0,       # use full (1.0) train set or partial (<1.0) train set
   "augmentation": "yes",       # whether to use data augmentation in data_loader
   "teacher": "densenet",       # no need to specify this for "base" cnn/resnet18
   "alpha": 0.0,                # only used for experiments involving distillation
   "temperature": 1,            # only used for experiments involving distillation
   "learning_rate": 1e-1,       # as the name suggests
   "batch_size": 128,           # for both train/eval
   "num_epochs": 200,           # as the name suggests
   "dropout_rate": 0.5,         # only valid for "cnn"-related models, not in resnet18
   "num_channels": 32,          # only valid for "cnn"-related models, not in resnet18
   "save_summary_steps": 100,
   "num_workers": 4

"""


import argparse
import os
from subprocess import check_call
import sys
import utils
import logging


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')

def launch_training_job(parent_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir}".format(python=PYTHON,
                                                             model_dir=model_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Set the logger
    utils.set_logger(os.path.join(args.parent_dir, 'search_hyperparameters.log'))

    '''
    Temperature and alpha search for KD on CNN (teacher model picked in params.json)
    Perform hypersearch (empirical grid): distilling 'temperature', loss weight 'alpha'
    '''

    # hyperparameters for KD
    alphas = [0.99, 0.95, 0.5, 0.1, 0.05]
    temperatures = [20., 10., 8., 6., 4.5, 3., 2., 1.5]

    logging.info("Searching hyperparameters...")
    logging.info("alphas: {}".format(alphas))
    logging.info("temperatures: {}".format(temperatures))

    for alpha in alphas:
        for temperature in temperatures:
            # [Modify] the relevant parameter in params (others remain unchanged)
            params.alpha = alpha
            params.temperature = temperature

            # Launch job (name has to be unique)
            job_name = "alpha_{}_Temp_{}".format(alpha, temperature)
            launch_training_job(args.parent_dir, job_name, params)