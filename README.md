# MixupExplainer

## DATASET

1. BA-Shapes
2. BA-Community
3. Tree-Circle
4. Tree-Grids
5. BA-2motifs
6. Mutag

## Explainer

1. GNNExplainer
2. PGExplainer
3. Mixup-GNNexplainer
4. Mixup-PGExplainer

Note: The MixupExplainer is implemented based on and refer to [https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks.git]

## ENVIRONMENT

## A QUICK START
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

