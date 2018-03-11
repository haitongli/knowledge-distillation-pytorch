"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


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

    # # Perform hypersearch over one parameter
    # learning_rates = [1e-4, 1e-3, 1e-2]

    # for learning_rate in learning_rates:
    #     # [Modify] the relevant parameter in params (others remain unchanged)
    #     params.learning_rate = learning_rate

    #     # Launch job (name has to be unique)
    #     job_name = "learning_rate_{}".format(learning_rate)
    #     launch_training_job(args.parent_dir, job_name, params)

    '''
    Temperature and alpha search for KD on CNN
    Perform hypersearch (grid): KD temperature, alpha
    '''
    alphas = [0.999, 0.95, 0.5, 0.1, 0.01]
    temperatures = [40, 20, 10, 8, 6, 4.5, 3, 2, 1]

    for alpha in alphas:
        for temperature in temperatures:
            # [Modify] the relevant parameter in params (others remain unchanged)
            params.alpha = alpha
            params.temperature = temperature

            # Launch job (name has to be unique)
            job_name = "alpha_{}_Temp_{}".format(alpha, temperature)
            launch_training_job(args.parent_dir, job_name, params)
   



