from math import sqrt
import random
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
import networkx as nx
from BaseExplainer import BaseExplainer
from graph import index_edge
from data_utils import adj_to_edge_index

"""
This is an adaption of the GNNExplainer of the PyTorch-Lightning library. 

The main similarity is the use of the methods _set_mask and _clear_mask to handle the mask. 
The main difference is the handling of different classification tasks. The original Geometric implementation only works for node 
classification. The implementation presented here also works for graph_classification datasets. 

link: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/gnn_explainer.html
"""


class RepExplainer(BaseExplainer):
    """
    An alt class encaptulating the MixUpExplainer.
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph"
    :param epochs: amount of epochs to train our explainer
    :param lr: learning rate used in the training of the explainer
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    
    :function __set_masks__: utility; sets learnable mask on the graphs.
    :function __clear_masks__: utility; rmoves the learnable mask.
    :function _loss: calculates the loss of the explainer
    :function explain: trains the explainer to return the subgraph which explains the classification of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, reg_coefs=(0.05, 1.0)):
        super().__init__(model_to_explain, graphs, features, task)
        self.epochs = epochs
        self.lr = lr
        self.bias = 0.0
        self.reg_coefs = reg_coefs
        self.temp = [5.0, 2.0]
        self.training = False
        self.noisy_graph = []

    def _set_masks(self, x, edge_index):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """

        (N, F), E = x.size(), edge_index.size(1)
        # print(N, F, E)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        self.edge_mask = None

    def forward(self, feats, graph, pred_label,
                graph2,
                reg_coefs=1, temperature=1.0, bias=0.0):

        if self.training:
            def sampling_mask(bias, edge_mask):
                bias = bias + 0.499  # If bias is 0, we run into problems
                # print('bias: ', bias)
                rand_ems = torch.rand(edge_mask.size())
                # print('rand_ems: ', rand_ems)
                eps = (bias - (1 - bias)) * rand_ems + (1 - bias)
                # print('eps: ', eps)
                gate_inputs = torch.log(eps) - torch.log(1 - eps)
                # print('gate inputs: ', gate_inputs)
                gate_inputs = (gate_inputs + edge_mask) / temperature
                # print('gate inputs: ', gate_inputs)
                mask_ = torch.sigmoid(gate_inputs)
                return mask_
            mask1 = sampling_mask(bias, self.edge_mask)
        else:
            mask1 = torch.sigmoid(self.edge_mask)

        mask = mask1
        masked_pred1, _ = self.model_to_explain(feats, graph2, edge_weights=mask)
        # masked_pred_2 = self.model_to_explain(feats, graph, edge_weights=mask)
        loss1 = self._loss(masked_pred1, pred_label, mask, self.reg_coefs)
        # print(self.edge_mask[:10])
        # print(torch.sigmoid(self.edge_mask[:20]))
        # print(loss)
        # print()
        return loss1

    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """

        # assert 0
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred) / 100
        # mape_loss = self.model_to_explain.mape(masked_pred, original_pred) / 100
        # print('mask pred: ', masked_pred)
        # print('original pred: ', original_pred)
        # cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        # print(mse_loss, size_loss, mask_ent_loss)
        return mse_loss + size_loss + mask_ent_loss

    def prepare(self, args):
        """Nothing is done to prepare the GNNExplainer, this happens at every index"""
        return

    def explain(self, index):
        """
        Main method to construct the explanation for a given sample. This is done by training a mask such that the masked graph still gives
        the same prediction as the original graph using an optimization approach
        :param index: index of the node/graph that we wish to explain
        :return: explanation graph and edge weights
        """
        index1 = int(index)
        while True:
            index2 = int(random.randint(0, len(self.graphs)-1))
            if index2 != index1:
                break
        # Prepare model for new explanation run
        self.model_to_explain.eval()
        self._clear_masks()

        feats1 = self.features[index1].detach()
        graph1 = self.graphs[index1].detach()
        feats2 = self.features[index2].detach()
        graph2 = self.graphs[index2].detach()
        # print(feats.shape, graph.shape)
        # Remove self-loops
        # graph = graph[:, (graph[0] != graph[1])]
        # print(feats.shape, graph.shape)
        with torch.no_grad():
            original_pred1, _ = self.model_to_explain(feats1, graph1)
            pred_label1 = original_pred1.detach()  # .argmax(dim=-1).detach()

        self._set_masks(feats1, graph1)
        # masked_pred = self.model_to_explain(feats, graph)
        # print('mask pred: ', masked_pred)
        # print('original pred: ', original_pred)
        # assert 0
        optimizer = Adam([self.edge_mask], lr=self.lr)
        # Start training loop
        self.training = True
        # print('start training...')
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))
        for e in range(0, self.epochs):
            optimizer.zero_grad()
            temperature = temp_schedule(e)
            # Sample possible explanation
            loss = self.forward(feats1, graph1, pred_label1,
                                graph2,
                                self.reg_coefs, temperature, self.bias)
            # print(e, loss)
            if e + 1 % 1000 == 0:
                print(e, loss)
                # assert 0
            loss.backward()
            optimizer.step()
        # assert 0

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        # print(mask)

        expl_graph_weights = torch.zeros(graph1.size(1))
        for i in range(len(mask)):  # Link explanation to original graph
            pair = graph1.T[i]
            t = index_edge(graph1, pair)
            expl_graph_weights[t] = mask[i]
        # print(mask[:20])
        # assert 0
        return graph1, expl_graph_weights

