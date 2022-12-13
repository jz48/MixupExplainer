from math import sqrt
import random
import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge

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
        self.reg_coefs = reg_coefs


    def _set_masks(self, x, edge_index):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """
        (N, F), E = x.size(), edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)


    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
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
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

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
            # Remove self-loops
            graph = graph[:, (graph[0] != graph[1])]
            with torch.no_grad():
                original_pred, _ = self.model_to_explain(feats, graph)
                pred_label = original_pred.argmax(dim=-1).detach()

        self._set_masks(feats, graph)
        optimizer = Adam([self.edge_mask], lr=self.lr)

        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()

            # Sample possible explanation
            if self.type == 'node':
                masked_pred, _ = self.model_to_explain(feats, graph)[index]
                loss = self._loss(masked_pred.unsqueeze(0), pred_label.unsqueeze(0), self.edge_mask, self.reg_coefs)
            else:
                masked_pred, _ = self.model_to_explain(feats, graph, edge_weights=torch.sigmoid(self.edge_mask))
                loss = self._loss(masked_pred, pred_label, self.edge_mask, self.reg_coefs)

            loss.backward()
            optimizer.step()

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, self.edge_mask.size(0)): # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights


class MixUpGNNExplainer(BaseExplainer):
    """
    A class encaptulating the GNNexplainer (https://arxiv.org/abs/1903.03894).
    with mixup graph generation
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
        self.dropoutlayer = nn.Dropout(0.1)

    def _set_masks(self, x1, edge_index1, x2, edge_index2):
        """
        Inject the explanation maks into the message passing modules.
        :param x: features
        :param edge_index: graph representation
        """

        mask1 = []
        mask2 = []
        edge_dict = {}
        merge_edge_index = [[], []]
        mark_idx_1 = 0
        mark_idx_2 = 0

        def check_smaller_index(a1_, a2_, b1_, b2_):
            if a1_ < b1_:
                return True
            if a1_ > b1_:
                return False
            if a1_ == b1_:
                if a2_ < b2_:
                    return True
                else:
                    return False
        while True:
            a1 = edge_index1[0][mark_idx_1].item()
            b1 = edge_index2[0][mark_idx_2].item()
            a2 = edge_index1[1][mark_idx_1].item()
            b2 = edge_index2[1][mark_idx_2].item()
            if a1 == b1 and a2 == b2:
                src = a1
                tgt = a2
                merge_edge_index[0].append(src)
                merge_edge_index[1].append(tgt)
                mask1.append(1)
                mask2.append(1)
                mark_idx_1 += 1
                mark_idx_2 += 1
            elif check_smaller_index(a1, a2, b1, b2):
                src = a1
                tgt = a2
                merge_edge_index[0].append(src)
                merge_edge_index[1].append(tgt)
                mask1.append(1)
                mask2.append(0)
                mark_idx_1 += 1
            else:
                src = b1
                tgt = b2
                merge_edge_index[0].append(src)
                merge_edge_index[1].append(tgt)
                mask1.append(0)
                mask2.append(1)
                mark_idx_2 += 1
            if mark_idx_1 >= len(edge_index1[0]):
                while mark_idx_2 < len(edge_index2[0]):
                    src = edge_index2[0][mark_idx_2].item()
                    tgt = edge_index2[1][mark_idx_2].item()
                    merge_edge_index[0].append(src)
                    merge_edge_index[1].append(tgt)
                    mask1.append(0)
                    mask2.append(1)
                    mark_idx_2 += 1
                break
            if mark_idx_2 >= len(edge_index2[0]):
                while mark_idx_1 < len(edge_index1[0]):
                    src = edge_index1[0][mark_idx_1].item()
                    tgt = edge_index1[1][mark_idx_1].item()
                    merge_edge_index[0].append(src)
                    merge_edge_index[1].append(tgt)
                    mask1.append(1)
                    mask2.append(0)
                    mark_idx_1 += 1
                break

        (N, F), E = x1.size(), len(merge_edge_index[0])

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.mask1 = torch.tensor(mask1)
        self.mask2 = torch.tensor(mask2)

        self.edge_mask1 = torch.nn.Parameter(torch.randn(E) * std)
        self.edge_mask1_masked = torch.mul(self.edge_mask1, self.mask1)
        self.edge_mask2 = torch.nn.Parameter(torch.randn(E) * std)
        self.edge_mask2_masked = torch.mul(self.edge_mask2, self.mask2)


        self.merge_edge_index = torch.tensor(merge_edge_index)
        self.delta = 0.5  # or a trainable parameter

        # self.edge_mask1 = torch.nn.Parameter(torch.randn(E) * std)
        pass
        # for module in self.model_to_explain.modules():
        #     if isinstance(module, MessagePassing):
        #         module.__explain__ = True
        #         module.__edge_mask__ = self.edge_mask

    def _clear_masks(self):
        """
        Cleans the injected edge mask from the message passing modules. Has to be called before any new sample can be explained.
        """
        self.edge_mask1 = None
        self.edge_mask2 = None

    def forward(self, feats1, graph1, pred_label1,
                feats2, graph2, pred_label2,
                temperature=1.0, bias=0.0):

        if 0:
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
            mask1 = sampling_mask(bias, self.edge_mask1_)
            mask2 = sampling_mask(bias, self.edge_mask2_)
        else:
            self.edge_mask1_masked = torch.mul(self.edge_mask1, self.mask1)
            self.edge_mask2_masked = torch.mul(self.edge_mask2, self.mask2)
            mask1 = torch.sigmoid(self.edge_mask1_masked)
            mask2 = torch.sigmoid(self.edge_mask2_masked)
            mask1 = torch.mul(mask1, torch.tensor(self.mask1))
            mask2 = torch.mul(mask2, torch.tensor(self.mask2))
            # print('mask1: ', mask1)
            # print('mask2: ', mask2)

        # print(mask1)
        # print(mask2)
        # t1 = torch.add(mask1, torch.tensor(self.mask2))

        # print(t1)
        # t2 = torch.mul(mask2, torch.tensor(-1))
        t2 = self.mask2 - mask2
        # print('t2: ', t2)
        t2 = self.dropoutlayer(t2)
        # print(t2)
        mask_pred1 = torch.add(mask1, t2)
        # print(mask_pred1)

        t3 = self.dropoutlayer(self.mask1 - mask1)
        mask_pred2 = torch.add(mask2, t3)
        # print(mask_pred1)
        # assert 0
        masked_pred1, _ = self.model_to_explain(feats1, self.merge_edge_index, edge_weights=mask_pred1)
        masked_pred2, _ = self.model_to_explain(feats2, self.merge_edge_index, edge_weights=mask_pred2)
        # masked_pred_2 = self.model_to_explain(feats, graph, edge_weights=mask)
        loss1 = self._loss(masked_pred1, pred_label1, mask1, self.reg_coefs)
        loss2 = self._loss(masked_pred2, pred_label2, mask2, self.reg_coefs)
        # print(self.edge_mask[:10])
        # print(torch.sigmoid(self.edge_mask[:20]))
        # print(loss)
        # print()
        return loss1 + loss2

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
        # mask = edge_mask
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask + EPS) - (1 - mask) * torch.log(1 - mask + EPS)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

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

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred, _ = self.model_to_explain(feats, graph)[index]
                pred_label = original_pred.argmax(dim=-1).detach()
        else:
            feats1 = self.features[index1].detach()
            graph1 = self.graphs[index1].detach()
            feats2 = self.features[index2].detach()
            graph2 = self.graphs[index2].detach()
            # Remove self-loops
            # graph = graph1[:, (graph1[0] != graph1[1])]
            with torch.no_grad():
                original_pred1, _ = self.model_to_explain(feats1, graph1)
                pred_label1 = original_pred1.argmax(dim=-1).detach()
                original_pred2, _ = self.model_to_explain(feats2, graph2)
                pred_label2 = original_pred2.argmax(dim=-1).detach()

        self._set_masks(feats1, graph1, feats2, graph2)
        optimizer = Adam([self.edge_mask1, self.edge_mask2], lr=self.lr)

        # Start training loop
        for e in range(0, self.epochs):
            optimizer.zero_grad()

            # Sample possible explanation
            if self.type == 'node':
                masked_pred = self.model_to_explain(feats, graph)[index]
                loss = self._loss(masked_pred.unsqueeze(0), pred_label.unsqueeze(0), self.edge_mask, self.reg_coefs)
            else:
                # masked_pred = self.model_to_explain(feats, graph)
                # loss = self._loss(masked_pred, pred_label, self.edge_mask, self.reg_coefs)
                loss = self.forward(feats1, graph1, pred_label1,
                                    feats2, graph2, pred_label2,
                                    self.reg_coefs)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask1)
        final_mask = []
        for i in range(len(self.mask1)):
            # print(self.mask1[i])
            if self.mask1[i] == 1:
                final_mask.append(mask[i])
        expl_graph_weights = torch.zeros(graph1.size(1))
        for i in range(len(final_mask)):  # Link explanation to original graph
            pair = graph1.T[i]
            t = index_edge(graph1, pair)
            expl_graph_weights[t] = final_mask[i]

        return graph1, expl_graph_weights

