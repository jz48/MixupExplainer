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


class GNNExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).
    
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

    def create_noisy_graph(self, size):
        if self.noisy_graph and 0:
            return self.noisy_graph[0], self.noisy_graph[1]
        graph = nx.barabasi_albert_graph(n=size, m=1)
        # print('original graph: ', graph)
        graph = graph.edges
        # print('edges: ', graph)
        features = []
        for _ in range(size):
            features.append([random.randrange(1, 1000) / 10] * 10)
        self.noisy_graph = [graph, features]
        return graph, features

    def forward(self, feats, graph, pred_label, reg_coefs, temperature=1.0, bias=0.0):
        b_graph, b_feat = self.create_noisy_graph(26)
        # print('1: ', len(b_feat), len(b_graph))
        # print('2: ', feats.shape, graph.shape)
        b_feat = torch.tensor(b_feat[25:])
        b_feat = torch.cat((feats, b_feat), dim=0)
        edge_index = [[], []]
        for i in range(len(graph[0])):
            edge_index[0].append(graph[0][i].item())
            edge_index[1].append(graph[1][i].item())
        for i in b_graph:
            if i[0] < 25 and i[1] < 25:
                pass
            else:
                edge_index[0].append(i[0])
                edge_index[1].append(i[1])
                edge_index[0].append(i[1])
                edge_index[1].append(i[0])
        edge_index = torch.tensor(edge_index)
        # print('3: ', b_feat.shape, edge_index.shape)

        # mask = torch.sigmoid(self.edge_mask)
        # masked_pred = self.model_to_explain(feats, graph, edge_weights=mask)

        # Regularization losses
        # mask = torch.sigmoid(edge_mask)
        # print('edge mask: ', self.edge_mask)
        if self.training:
            bias = bias + 0.499  # If bias is 0, we run into problems
            # print('bias: ', bias)
            rand_ems = torch.rand(self.edge_mask.size())
            # print('rand_ems: ', rand_ems)
            eps = (bias - (1 - bias)) * rand_ems + (1 - bias)
            # print('eps: ', eps)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            # print('gate inputs: ', gate_inputs)
            gate_inputs = (gate_inputs + self.edge_mask) / temperature
            # print('gate inputs: ', gate_inputs)
            mask = torch.sigmoid(gate_inputs)
        else:
            mask = torch.sigmoid(self.edge_mask)
        # print('mask after sampling: ', mask)
        # print('mask: ', mask)
        # print(edge_index.shape[1], mask.shape[0])
        padding = torch.ones(edge_index.shape[1] - mask.shape[0], requires_grad=False)
        # padding /= 10
        # print(padding)
        n_edge_mask = torch.cat((mask, padding))
        # print(n_edge_mask)
        # assert 0
        masked_pred, _ = self.model_to_explain(b_feat, edge_index, edge_weights=n_edge_mask)
        # masked_pred_2 = self.model_to_explain(feats, graph, edge_weights=mask)
        loss = self._loss(masked_pred, pred_label, mask, self.reg_coefs)
        # print(self.edge_mask[:10])
        # print(torch.sigmoid(self.edge_mask[:20]))
        # print(loss)
        # print()
        return loss

    def forward2(self, feats, graph, pred_label, reg_coefs, temperature=1.0, bias=0.0):

        if self.training:
            bias = bias + 0.499  # If bias is 0, we run into problems
            # print('bias: ', bias)
            rand_ems = torch.rand(self.edge_mask.size())
            # print('rand_ems: ', rand_ems)
            eps = (bias - (1 - bias)) * rand_ems + (1 - bias)
            # print('eps: ', eps)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            # print('gate inputs: ', gate_inputs)
            gate_inputs = (gate_inputs + self.edge_mask) / temperature
            # print('gate inputs: ', gate_inputs)
            mask = torch.sigmoid(gate_inputs)
        else:
            mask = torch.sigmoid(self.edge_mask)
        # print('mask after sampling: ', mask)
        masked_pred_2 = self.model_to_explain(feats, graph, edge_weights=mask)
        loss = self._loss(masked_pred_2, pred_label, mask, self.reg_coefs)
        # print(self.edge_mask[:10])
        # print(torch.sigmoid(self.edge_mask[:20]))
        # print(loss)
        # print()
        return loss

    def disturb_loss(self, feats, graph, pred_label):
        loss = 0
        mask = self.edge_mask

        return loss

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
        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.eval()
        self._clear_masks()
        self.noisy_graph = []
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred, _ = self.model_to_explain(feats, graph)[index]
                pred_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            graph = self.graphs[index].detach()
            # print(feats.shape, graph.shape)
            # Remove self-loops
            # graph = graph[:, (graph[0] != graph[1])]
            # print(feats.shape, graph.shape)
            with torch.no_grad():
                original_pred, _ = self.model_to_explain(feats, graph)
                pred_label = original_pred.detach()  # .argmax(dim=-1).detach()

        self._set_masks(feats, graph)
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
            loss = self.forward(feats, graph, pred_label, self.reg_coefs, temperature, self.bias)
            print(e, loss)
            if e + 1 % 1000 == 0:
                print(e, loss)
                assert 0
            loss.backward()
            optimizer.step()
        # assert 0

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)

        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, self.edge_mask.size(0)):  # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]
        # print(mask[:20])
        return graph, expl_graph_weights


class ExdGNNExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).
    extended version
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
        self.reg_coefs = reg_coefs

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
        # print(self.edge_mask.shape)
        # assert 0
        for module in self.model_to_explain.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        for module in self.model_to_explain.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.edge_mask = None

    def _loss(self, masked_pred, original_pred, edge_mask, reg_coefs):
        """
        Returns the loss score based on the given mask.
        :param masked_pred: Prediction based on the current explanation
        :param original_pred: Predicion based on the original graph
        :param edge_mask: Current explanaiton
        :param reg_coefs: regularization coefficients
        :return: loss
        """
        size_reg = reg_coefs[0]
        entropy_reg = reg_coefs[1]
        EPS = 1e-15

        # Regularization losses
        mask = torch.sigmoid(edge_mask)
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        mse_loss = torch.nn.functional.mse_loss(masked_pred, original_pred)

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
        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.eval()
        self._clear_masks()

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)[index]
                pred_label = original_pred.argmax(dim=-1).detach()
        else:
            feats = self.features[index].detach()
            graph = self.graphs[index].detach()
            # print(feats.shape, graph.shape)
            # Remove self-loops
            # graph = graph[:, (graph[0] != graph[1])]
            # print(feats.shape, graph.shape)
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)
                pred_label = original_pred.detach()  # .argmax(dim=-1).detach()

        self._set_masks(feats, graph)
        # masked_pred = self.model_to_explain(feats, graph)
        # print('mask pred: ', masked_pred)
        # print('original pred: ', original_pred)
        # assert 0
        optimizer = Adam([self.edge_mask], lr=self.lr)

        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()

            # Sample possible explanation
            if self.type == 'node':
                masked_pred = self.model_to_explain(feats, graph)[index]
                loss = self._loss(masked_pred.unsqueeze(0), pred_label.unsqueeze(0), self.edge_mask, self.reg_coefs)
            else:
                mask = torch.sigmoid(self.edge_mask)
                if e > 8000 and 0:
                    print(self.edge_mask)
                    print(mask)
                    assert 0
                masked_pred = self.model_to_explain(feats, graph, edge_weights=mask)
                loss = self._loss(masked_pred, pred_label, self.edge_mask, self.reg_coefs)
            # print(self.edge_mask[:10])
            # print(torch.sigmoid(self.edge_mask[:20]))
            # print(loss)
            # print()
            loss.backward()
            optimizer.step()

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)

        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, self.edge_mask.size(0)):  # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]
        # print(mask[:20])
        return graph, expl_graph_weights


