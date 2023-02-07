# MixupExplainer

## DATASET

## ENVIRONMENT

## a quick start

1. train the gnn model
```
cd MixupExplainer
python experiment_models_training.py ba2motif gnn
```

The two command line args mean: 1. the dataset trained on. 2. The graph model we use.

2. run the explainer
```
python exd_explainer.py 3
```
'3' means using the MixUpExplainer, support to other explainers will be added later.


3. The MixupExplainer is implemented based on and refer to [https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks.git]
