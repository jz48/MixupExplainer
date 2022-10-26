#!/usr/bin/env python
# coding: utf-8
import sys

# # Model training
# 
# This notebook can be used to train the models used for the replication experiment. The notebook makes heavy use of predefined configuration files that describe the parameter setting of each model. Pretrained models using these specific parameters are also already available. Hence, retraining the models is not needed if you only wish to replicate the replication experiment. 
# 
# If you do wish to perform the replication experiments with your own retrained models, it is not sufficient to only retrain the model with this script. To prevent the training script from accidentally overriding the pretrained models, the models are saved in a different location then where the pretrained models are loaded from. 
# 
# **To replace the pretrained models in the replication study** you therefore need to copy the trained model from `checkpoints` to `Explanation/models/pretrained/<_model>/<_dataset>`. Where \_model and \_dataset are defined as in the code below. 

# In[1]:


from selector import Selector
from training import train_node, train_graph

import torch
import numpy as np

# In[2]:


# _dataset = 'bareg'  # One of: bashapes, bacommunity, treecycles, treegrids, ba2motifs, mutag
# _dataset = 'bareg2'
# _dataset = 'bareg3'
_dataset = 'bareg1'
_dataset = sys.argv[1]
# Parameters below should only be changed if you want to run any of the experiments in the supplementary
_folder = 'replication'  # One of: replication, batchnorm
# _model = 'gnn' if _folder == 'replication' else 'ori'
_model = 'reg'  # or gnn
_model = sys.argv[2]
# PGExplainer
config_path = f"./configs/{_folder}/models/model_{_model}_{_dataset}.json"

config = Selector(config_path)
extension = (_folder == 'extension')

# In[3]:


config = Selector(config_path).args

torch.manual_seed(config.model.seed)
torch.cuda.manual_seed(config.model.seed)
np.random.seed(config.model.seed)

# In[4]:


_dataset = config.model.dataset
_explainer = config.model.paper

if _dataset[:3] == "syn":
    train_node(_dataset, _explainer, config.model)
elif _dataset == "mutag" or _dataset.startswith('bareg'):
    train_graph(_dataset, _explainer, config.model)

# In[ ]:
