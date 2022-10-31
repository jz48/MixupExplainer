# XAI_GNN_regression

The ExplanationEvaluation folder contains original code and the remake folder contains new code.

## a quick start

1. train the gnn model
```
cd XAI_GNN_regression/remake
python experiment_models_training.py bareg1 gnn
```

The two command line args mean: 1. the dataset trained on. 2. The graph model we use.

2. run the explainer
```
python exd_explainer.py
```
