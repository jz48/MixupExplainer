import torch
import os

from gnn_models import NodeGCN, GraphGCN, RegGNN


def string_to_model(paper, dataset):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.
    :param paper: the paper who's classification model we want to use.
    :param dataset: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    :returns: torch.nn.module models
    """
    print(paper)
    if paper == "GNN":
        if dataset == "bareg1":
            return GraphGCN(10, 200)
        elif dataset == "bareg2":
            return GraphGCN(10, 20)
        elif dataset == "mutag":
            return GraphGCN(14, 2)
        else:
            raise NotImplementedError
    elif paper == "REG":
        if dataset == "bareg1":
            return RegGNN(25, 10, 20, 200, 0.2)
        elif dataset == "bareg2":
            return RegGNN(120, 10, 20, 20, 0.2)
        elif dataset == "mutag":
            return RegGNN(14, 2)
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
    path = f"{dir_path}/checkpoints/{paper}/{dataset}/best_model"
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
            print(key)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            for key in checkpoint['model_state_dict']:
                if key.startswith('conv') and key.endswith('weight'):
                    new_key = key[:5] + key[-7:]
                    renamed_state_dict[new_key] = (checkpoint['model_state_dict'][key])
                    # renamed_state_dict[new_key] = (checkpoint['model_state_dict'][key]).T
                else:
                    renamed_state_dict[key] = checkpoint['model_state_dict'][key]
            model.load_state_dict(renamed_state_dict)
        # model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f"This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
        if return_checkpoint:
            return model, checkpoint
    return model
