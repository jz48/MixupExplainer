import os.path

import torch.optim as optim
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
from selector import Selector
from replication import replication


def train_explainer(config):
    _dataset = 'bareg1'
    _model = 'gnn'
    _explainer = 'mixupexplainer'
    _folder = 'replication'
    config_path = f"/data/jiaxing/xai/XAI_GNN_regression/remake/configs/{_folder}/explainers/{_explainer}/{_dataset}_{_model}_7.json"
    print('config_path: ', config_path)
    print(os.path.exists(config_path))
    print(Selector(config_path))
    o_config = {
        "config_name": "bareg1_gnn_7",
        "dataset": "bareg1",
        "model": "GNN",
        "explainer": "MixUp",
        "lr": config['lr'],
        "epochs": config['epochs'],
        "sample_bias": 0.0,
        "reg_size": 0.2,
        "reg_ent": 0.01,
        "temps": [5.0, 1.0],
        "seeds": [0],
        "eval_enabled": True,
        "thres_snip": 5,
        "thres_min": -1
    }

    print(o_config['lr'], o_config['epochs'])
    # (auc, auc_std), inf_time = replication(config.args.explainer, False)

    for i in range(10):
        (auc, auc_std), inf_time = replication(config, False)
        acc = auc
        tune.report(mean_accuracy=acc)


if __name__ == '__main__':
    analysis = tune.run(
        train_explainer,
        config={"lr": tune.grid_search([0.00001, 0.00005]),
                "epochs": tune.grid_search([2000, 1000])})
    '''
    analysis = tune.run(
        train_explainer, config={"lr": tune.grid_search([0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
                                 "epochs": tune.grid_search([10000, 8000, 5000, 2000])})
    '''
    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
