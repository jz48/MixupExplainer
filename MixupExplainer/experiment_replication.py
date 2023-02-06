#!/usr/bin/env python
# coding: utf-8
import sys
from ray import tune

# # Replication experiment
# 
# This is the main notebook required to obtain the results of our replication study. The notebook is build around the concepts of predefined configuration files. These configuration files can be found within the codebase. The configuration files for different datasets and different explainers can be chosen by changing the parameters in the second codeblock. 
# 
# When loaded, the configuration for a replication experiment is passed to the replication function. This function is responsible for running all parts of the evaluation; quantitative, qualitative and efficiency. The results for the quantitative and efficiency studies are returned by the replication method and also stored in the `results` folder. The results of the qualitative study are stored in the folder name `qualitative`. 
# 
# **Be aware that the replication function can take very long to completed**. This is caused by the method averaging all scores over ten runs. If speed is required over accuracy the last line of the 2nd codeblock can be uncommented. This will make the evaluation run over one run only. 
# 

# In[1]:


from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication

# In[ ]:

dataset_mark = int(sys.argv[1])

if dataset_mark == 1:
    _dataset = 'bashapes'  # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
elif dataset_mark == 2:
    _dataset = 'bacommunity'
elif dataset_mark == 3:
    _dataset = 'treecycles'
elif dataset_mark == 4:
    _dataset = 'treegrids'
elif dataset_mark == 5:
    _dataset = 'ba2motifs'
elif dataset_mark == 5:
    _dataset = 'mutag'
else:
    _dataset = 'ba2motifs'

model_mark = int(sys.argv[2])

if model_mark == 1:
    _explainer = 'gnnexplainer'  # One of: pgexplainer, gnnexplainer
elif model_mark == 2:
    _explainer = 'pgexplainer'
elif model_mark == 3:
    _explainer = 'mixupgnnexplainer'
elif model_mark == 4:
    _explainer = 'mixuppgexplainer'
elif model_mark == 5:
    _explainer = 'mixupsftgnnexplainer'
elif model_mark == 6:
    _explainer = 'mixupsftpgexplainer'
else:
    _explainer = 'gnnexplainer'

print(_dataset, _explainer)
# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication'  # One of: replication, extension

# PGExplainer
config_path = f"./ExplanationEvaluation/configs/{_folder}/explainers/{_explainer}/{_dataset}.json"

config0 = Selector(config_path)
extension = (_folder == 'extension')

# config.args.explainer.seeds = [0]


def train_explainer(config):
    # _dataset = 'bareg1'
    # _model = 'gnn'
    # _explainer = 'mixupgnnexplainer'
    # _folder = 'replication'
    # config_path = f"/data/jiaxing/xai/XAI_GNN_regression/remake/configs/{_folder}/explainers/{_explainer}/{_dataset}_{_model}_7.json"
    print('config_path: ', config_path)
    # print(os.path.exists(config_path))
    o_config = config0.args.explainer
    print(o_config)
    print(o_config.lr, o_config.epochs)
    # (auc, auc_std), inf_time = replication(config.args.explainer, False)
    # o_config.lr = config['lr']
    # o_config.epochs = config['epochs']
    o_config.yita = config['yita']
    for i in range(1):
        # (auc, auc_std), inf_time = replication(config.args.explainer, extension)
        (auc, auc_std), inf_time = replication(o_config, extension)
        print((auc, auc_std), inf_time)
        acc = auc
        tune.report(mean_accuracy=acc)


if __name__ == '__main__':
    do_analysis = False
    if do_analysis:
        analysis = tune.run(
            train_explainer, config={"lr": tune.grid_search([0.003]),
                                     "epochs": tune.grid_search([100]),
                                     "reg_size": tune.grid_search([0.05]),
                                     "reg_ent": tune.grid_search([1.0]),
                                     "yita": tune.grid_search([300, 1000, 3000, 5000, 7000, 10000, 30000])})
        print("Best config: ", analysis.get_best_config(metric="mean_accuracy", mode='max'))

        # Get a dataframe for analyzing trial results.
        df = analysis.dataframe()
    else:
        config0.args.explainer.yita = 5
        (auc, auc_std), inf_time = replication(config0.args.explainer, extension)
        print((auc, auc_std), inf_time)
