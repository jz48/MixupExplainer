import torch
import os

from ExplanationEvaluation.models.GNN_paper import NodeGCN as GNN_NodeGCN
from ExplanationEvaluation.models.GNN_paper import GraphGCN as GNN_GraphGCN
from ExplanationEvaluation.models.GNN_paper import RegGNN as GNN_RegGNN
from ExplanationEvaluation.models.PG_paper import NodeGCN as PG_NodeGCN
from ExplanationEvaluation.models.PG_paper import GraphGCN as PG_GraphGCN

def string_to_model(paper, dataset):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    if paper == "GNN":
        if dataset in ['syn1']:
            return GNN_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return GNN_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return GNN_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return GNN_NodeGCN(10, 2)
        elif dataset == "ba2":
            return GNN_GraphGCN(10, 2)
        elif dataset == "bareg":
            return GNN_RegGNN(1, 20, 21, 0.2)
            # return GNN_GraphGCN(1, 200)
        elif dataset == "bareg2":
            # return GNN_GraphGCN(10, 21)
            return GNN_RegGNN(10, 20, 21, 0.2)
        elif dataset == "bareg2_2":
            # return GNN_GraphGCN(10, 21)
            return GNN_RegGNN(10, 20, 21, 0.2)
        elif dataset == "bareg3":
            # return GNN_GraphGCN(10, 100)
            return GNN_RegGNN(10, 20, 100, 0.2)
        elif dataset == "mutag":
            return GNN_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper == "PG":
        if dataset in ['syn1']:
            return PG_NodeGCN(10, 4)
        elif dataset in ['syn2']:
            return PG_NodeGCN(10, 8)
        elif dataset in ['syn3']:
            return PG_NodeGCN(10, 2)
        elif dataset in ['syn4']:
            return PG_NodeGCN(10, 2)
        elif dataset == "ba2":
            return PG_GraphGCN(10, 2)
        elif dataset == "mutag":
            return PG_GraphGCN(14, 2)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_pretrained_path(paper, dataset):
    """
    Given a paper and dataset loads the pre-trained model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: str; the path to the pre-trined model parameters.
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = f"{dir_path}/pretrained/{paper}/{dataset}/best_model"
    return path


def model_selector(paper, dataset, pretrained=True, return_checkpoint=False):
    """
    Given a paper and dataset loads accociated model.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :param pretrained: whter to return a pre-trained model or not.
    :param return_checkpoint: wheter to return the dict contining the models parameters or not.
    :returns: torch.nn.module models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset)
    if pretrained:
        path = get_pretrained_path(paper, dataset)
        print(type(model))
        print('load model from path: ', path)
        checkpoint = torch.load(path)
        renamed_state_dict = {}
        for key in checkpoint['model_state_dict']:
            if key.startswith('conv') and key.endswith('weight'):
                new_key = key[:5] + '.lin' + key[5:]
                renamed_state_dict[new_key] = (checkpoint['model_state_dict'][key]).T
            else:
                renamed_state_dict[key] = checkpoint['model_state_dict'][key]
        model.load_state_dict(renamed_state_dict)
        # model.load_state_dict(checkpoint['model_state_dict'])
        print(f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model