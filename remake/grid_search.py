import os.path

import torch.optim as optim
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test
from selector import Selector
from replication import replication


def train_explainer(config):
    _dataset = 'bareg1'
    _model = 'gnn'
    _explainer = 'mixupgnnexplainer'
    _folder = 'replication'
    config_path = f"/data/jiaxing/xai/XAI_GNN_regression/remake/configs/{_folder}/explainers/{_explainer}/{_dataset}_{_model}_7.json"
    print('config_path: ', config_path)
    print(os.path.exists(config_path))
    o_config = Selector(config_path).args.explainer
    print(o_config)
    print(o_config.lr, o_config.epochs)
    # (auc, auc_std), inf_time = replication(config.args.explainer, False)
    o_config.lr = config['lr']
    o_config.epochs = config['epochs']
    for i in range(10):
        (auc, auc_std), inf_time = replication(o_config, False)
        acc = auc
        tune.report(mean_accuracy=acc)


if __name__ == '__main__':
    analysis = tune.run(
        train_explainer, config={"lr": tune.grid_search([0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
                                 "epochs": tune.grid_search([10000, 8000, 5000, 2000])})
    print("Best config: ", analysis.get_best_config(metric="mean_accuracy", mode='max'))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
