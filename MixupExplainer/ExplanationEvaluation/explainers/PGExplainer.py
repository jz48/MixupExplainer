import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import random
import numpy as np
from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge


class PGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).
    
    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.
    
    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias

        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

    def _create_explainer_input(self, pair, embeds, node_id):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        if self.type == 'node':
            node_embed = embeds[node_id].repeat(rows.size(0), 1)
            input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
        else:
            # Node id is not used in this case
            input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
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

        # Regularization losses
        # scale = 0.99
        # mask = mask * (2 * scale - 1.0) + (1.0 - scale)
        # check later whether mask value is 0 or 1, which will cause error
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None:  # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.train(indices=indices)

    def train(self, indices=None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                if self.type == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()

                # Sample possible explanation
                input_expl = self._create_explainer_input(graph, embeds, n).unsqueeze(0)
                sampling_weights = self.explainer_model(input_expl)
                mask = self._sample_graph(sampling_weights, t, bias=self.sample_bias).squeeze()

                masked_pred, _ = self.model_to_explain(feats, graph, edge_weights=mask)
                original_pred, _ = self.model_to_explain(feats, graph)

                if self.type == 'node':  # we only care for the prediction of the node
                    masked_pred = masked_pred[n].unsqueeze(dim=0)
                    original_pred = original_pred[n]

                id_loss = self._loss(masked_pred, torch.argmax(original_pred).unsqueeze(0), mask, self.reg_coefs)
                loss += id_loss

            loss.backward()
            print(e, loss)
            optimizer.step()

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        index = int(index)
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.graphs[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

        # Use explainer mlp to get an explanation
        input_expl = self._create_explainer_input(graph, embeds, index).unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()

        expl_graph_weights = torch.zeros(graph.size(1))  # Combine with original graph
        for i in range(0, mask.size(0)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]

        return graph, expl_graph_weights


class MixUpPGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.

    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.dropoutlayer = nn.Dropout(0.1)
        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

    def _create_explainer_input(self, pair1, embeds1, node_id1, pair2, embeds2, node_id2):
        """
        Given the embeddign of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """

        mask1 = []
        mask2 = []
        merge_edge_index = [[], []]
        mark_idx_1 = 0
        mark_idx_2 = 0
        edge_index1 = pair1
        edge_index2 = pair2

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

        def build_input_expl(edge_index, embeds, node_id, mask):
            rows = edge_index[0]
            cols = edge_index[1]

            row_embeds = embeds[rows]
            col_embeds = embeds[cols]

            zeros = torch.zeros(len(embeds[0]))
            for i in range(len(row_embeds)):
                if mask[i] == 0:
                    row_embeds[i] = zeros.clone()
            for i in range(len(col_embeds)):
                if mask[i] == 0:
                    col_embeds[i] = zeros.clone()
            if self.type == 'node':
                node_embed = embeds[node_id].repeat(rows.size(0), 1)
                input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
            else:
                # Node id is not used in this case
                input_expl = torch.cat([row_embeds, col_embeds], 1)
            return input_expl

        self.mask1 = torch.tensor(mask1)
        self.mask2 = torch.tensor(mask2)
        self.merged_graph = torch.tensor(merge_edge_index)
        input_expl1 = build_input_expl(merge_edge_index, embeds1, node_id1, mask1)
        input_expl2 = build_input_expl(merge_edge_index, embeds2, node_id2, mask2)
        return input_expl1, input_expl2

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
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

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None:  # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.train(indices=indices)

    def train(self, indices=None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                while True:
                    n2 = int(random.randint(0, len(self.graphs) - 1))
                    if n2 != n:
                        break
                if self.type == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats = self.features
                    graph = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                else:
                    feats = self.features[n].detach()
                    graph = self.graphs[n].detach()
                    embeds = self.model_to_explain.embedding(feats, graph).detach()

                    feats2 = self.features[n2].detach()
                    graph2 = self.graphs[n2].detach()
                    embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()

                # Sample possible explanation
                input_expl1, input_expl2 = self._create_explainer_input(graph, embeds, n, graph2, embeds2, n2)
                input_expl1 = input_expl1.unsqueeze(0)
                input_expl2 = input_expl2.unsqueeze(0)
                sampling_weights1 = self.explainer_model(input_expl1)
                # sampling_weights1 = torch.mul(sampling_weights1, self.mask1)
                sampling_weights2 = self.explainer_model(input_expl2)
                # sampling_weights2 = torch.mul(sampling_weights2, self.mask2)
                mask1 = self._sample_graph(sampling_weights1, t, bias=self.sample_bias).squeeze()
                mask2 = self._sample_graph(sampling_weights2, t, bias=self.sample_bias).squeeze()
                mask1 = torch.mul(mask1, self.mask1)
                mask2 = torch.mul(mask2, self.mask2)

                t2 = self.dropoutlayer(self.mask2 - mask2)
                mask_pred1 = torch.add(mask1, t2)
                mask_pred1 = torch.sigmoid(mask_pred1)
                t3 = self.dropoutlayer(self.mask1 - mask1)
                mask_pred2 = torch.add(mask2, t3)
                mask_pred2 = torch.sigmoid(mask_pred2)
                masked_pred1, _ = self.model_to_explain(feats, self.merged_graph, edge_weights=mask_pred1)
                masked_pred2, _ = self.model_to_explain(feats2, self.merged_graph, edge_weights=mask_pred2)
                original_pred, _ = self.model_to_explain(feats, graph)

                if self.type == 'node':  # we only care for the prediction of the node
                    masked_pred = masked_pred[n].unsqueeze(dim=0)
                    original_pred = original_pred[n]

                id_loss = self._loss(masked_pred1, torch.argmax(original_pred).unsqueeze(0), mask_pred1, self.reg_coefs)
                loss += id_loss
                id_loss = self._loss(masked_pred2, torch.argmax(original_pred).unsqueeze(0), mask_pred2, self.reg_coefs)
                loss += id_loss
            loss.backward()
            optimizer.step()

    def explain(self, index):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        index = int(index)
        while True:
            index2 = int(random.randint(0, len(self.graphs) - 1))
            if index2 != index:
                break
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()
        else:
            feats = self.features[index].clone().detach()
            graph = self.graphs[index].clone().detach()
            embeds = self.model_to_explain.embedding(feats, graph).detach()

            feats2 = self.features[index2].detach()
            graph2 = self.graphs[index2].detach()
            embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()

        # Use explainer mlp to get an explanation
        input_expl, _ = self._create_explainer_input(graph, embeds, index, graph2, embeds2, index2)
        input_expl = input_expl.unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        final_mask = []
        for i in range(len(self.mask1)):
            # print(self.mask1[i])
            if self.mask1[i] == 1:
                final_mask.append(mask[i])
        expl_graph_weights = torch.zeros(graph.size(1))  # Combine with original graph
        for i in range(0, len(final_mask)):
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = final_mask[i]

        return graph, expl_graph_weights


class MixUpSFTPGExplainer(BaseExplainer):
    """
    A class encaptulating the PGExplainer (https://arxiv.org/abs/2011.04573).

    :param model_to_explain: graph classification model who's predictions we wish to explain.
    :param graphs: the collections of edge_indices representing the graphs.
    :param features: the collcection of features for each node in the graphs.
    :param task: str "node" or "graph".
    :param epochs: amount of epochs to train our explainer.
    :param lr: learning rate used in the training of the explainer.
    :param temp: the temperture parameters dictacting how we sample our random graphs.
    :param reg_coefs: reguaization coefficients used in the loss. The first item in the tuple restricts the size of the explainations, the second rescticts the entropy matrix mask.
    :params sample_bias: the bias we add when sampling random graphs.

    :function _create_explainer_input: utility;
    :function _sample_graph: utility; sample an explanatory subgraph.
    :function _loss: calculate the loss of the explainer during training.
    :function train: train the explainer
    :function explain: search for the subgraph which contributes most to the clasification decision of the model-to-be-explained.
    """

    def __init__(self, model_to_explain, graphs, features, task, epochs=30, lr=0.003, temp=(5.0, 2.0),
                 reg_coefs=(0.05, 1.0), sample_bias=0):
        super().__init__(model_to_explain, graphs, features, task)

        self.epochs = epochs
        self.lr = lr
        self.temp = temp
        self.reg_coefs = reg_coefs
        self.sample_bias = sample_bias
        self.dropoutlayer = nn.Dropout(0.1)
        if self.type == "graph":
            self.expl_embedding = self.model_to_explain.embedding_size * 2
        else:
            self.expl_embedding = self.model_to_explain.embedding_size * 3

    def _create_explainer_input(self, pair1, embeds1, node_id1, pair2, embeds2, node_id2):
        """
        Given the embedding of the sample by the model that we wish to explain, this method construct the input to the mlp explainer model.
        Depending on if the task is to explain a graph or a sample, this is done by either concatenating two or three embeddings.
        :param pair: edge pair
        :param embeds: embedding of all nodes in the graph
        :param node_id: id of the node, not used for graph datasets
        :return: concatenated embedding
        """

        mask1 = []
        mask2 = []
        edge_index1 = pair1
        edge_index2 = pair2

        def adj2edge_list(adj):
            edge_list = [[], []]
            for row in range(adj.shape[0]):
                for col in range(adj.shape[1]):
                    if adj[row][col] == 1:
                        edge_list[0].append(row)
                        edge_list[1].append(col)
            return edge_list

        def edge_list2adj(edge_list, size_adj):
            # adj = to_scipy_sparse_matrix(edge_list)
            adj = np.zeros((size_adj, size_adj), dtype=float)
            for i in range(len(edge_list[0])):
                adj[edge_list[0][i]][edge_list[1][i]] = 1.0
            return adj

        self.index1 = self.index1
        self.index2 = self.index2 + embeds1.size(0)

        adj1 = edge_list2adj(edge_index1, embeds1.size(0))
        adj2 = edge_list2adj(edge_index2, embeds2.size(0))

        link_adj1 = np.zeros((adj1.shape[0], adj2.shape[0]))
        yita = self.yita / (adj1.shape[0] * adj2.shape[0])
        # yita = 0.03
        for i in range(link_adj1.shape[0]):
            for j in range(link_adj1.shape[1]):
                if random.random() < yita:
                    link_adj1[i][j] = 1.0

        link_adj2 = link_adj1.T

        a = np.concatenate((adj1, link_adj1), axis=1)
        b = np.concatenate((link_adj2, adj2), axis=1)

        merged_adj = np.concatenate((a, b), axis=0)

        merge_edge_index = adj2edge_list(merged_adj)
        merged_feats = torch.cat((self.feats1, self.feats2), 0)
        merged_embeds = torch.cat((embeds1, embeds2), 0)
        for i in range(len(merge_edge_index[0])):
            if merge_edge_index[0][i] < embeds1.size(0) and merge_edge_index[1][i] < embeds1.size(0):
                mask1.append(1)
            else:
                mask1.append(0)
            if merge_edge_index[0][i] >= embeds1.size(0) or merge_edge_index[1][i] >= embeds1.size(0):
                mask2.append(1)
            else:
                mask2.append(0)

        def build_input_expl(edge_index, embeds, node_id, mask):
            rows = edge_index[0]
            cols = edge_index[1]

            row_embeds = embeds[rows]
            col_embeds = embeds[cols]

            zeros = torch.zeros(len(embeds[0]))
            for i in range(len(row_embeds)):
                if mask[i] == 0:
                    row_embeds[i] = zeros.clone()
            for i in range(len(col_embeds)):
                if mask[i] == 0:
                    col_embeds[i] = zeros.clone()
            if self.type == 'node':
                node_embed = embeds[node_id].repeat(len(rows), 1)
                input_expl = torch.cat([row_embeds, col_embeds, node_embed], 1)
            else:
                # Node id is not used in this case
                input_expl = torch.cat([row_embeds, col_embeds], 1)
            return input_expl

        self.mask1 = torch.tensor(mask1)
        self.mask2 = torch.tensor(mask2)
        self.merged_feats = merged_feats
        self.merged_graph = torch.tensor(merge_edge_index)
        input_expl1 = build_input_expl(merge_edge_index, merged_embeds, self.index1, mask1)
        input_expl2 = build_input_expl(merge_edge_index, merged_embeds, self.index2, mask2)
        return input_expl1, input_expl2

    def _sample_graph(self, sampling_weights, temperature=1.0, bias=0.0, training=True):
        """
        Implementation of the reparamerization trick to obtain a sample graph while maintaining the posibility to backprop.
        :param sampling_weights: Weights provided by the mlp
        :param temperature: annealing temperature to make the procedure more deterministic
        :param bias: Bias on the weights to make samplign less deterministic
        :param training: If set to false, the samplign will be entirely deterministic
        :return: sample graph
        """
        if training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(sampling_weights.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = (gate_inputs + sampling_weights) / temperature
            graph = torch.sigmoid(gate_inputs)
        else:
            graph = torch.sigmoid(sampling_weights)
        return graph

    def _loss(self, masked_pred, original_pred, mask, reg_coefs):
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

        # Regularization losses
        size_loss = torch.sum(mask) * size_reg
        mask_ent_reg = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
        mask_ent_loss = entropy_reg * torch.mean(mask_ent_reg)

        # Explanation loss
        cce_loss = torch.nn.functional.cross_entropy(masked_pred, original_pred)

        return cce_loss + size_loss + mask_ent_loss

    def prepare(self, indices=None):
        """
        Before we can use the explainer we first need to train it. This is done here.
        :param indices: Indices over which we wish to train.
        """
        # Creation of the explainer_model is done here to make sure that the seed is set
        self.explainer_model = nn.Sequential(
            nn.Linear(self.expl_embedding, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        if indices is None:  # Consider all indices
            indices = range(0, self.graphs.size(0))

        self.train(indices=indices)

    def train(self, indices=None):
        """
        Main method to train the model
        :param indices: Indices that we want to use for training.
        :return:
        """
        # Make sure the explainer model can be trained
        self.explainer_model.train()

        # Create optimizer and temperature schedule
        optimizer = Adam(self.explainer_model.parameters(), lr=self.lr)
        temp_schedule = lambda e: self.temp[0] * ((self.temp[1] / self.temp[0]) ** (e / self.epochs))

        # If we are explaining a graph, we can determine the embeddings before we run
        if self.type == 'node':
            embeds = self.model_to_explain.embedding(self.features, self.graphs).detach()

        # Start training loop
        for e in tqdm(range(0, self.epochs)):
            optimizer.zero_grad()
            loss = torch.FloatTensor([0]).detach()
            t = temp_schedule(e)

            for n in indices:
                n = int(n)
                self.index1 = n
                while True:
                    n2 = random.choice(indices)
                    if n2 != n:
                        self.index2 = n2
                        break
                if self.type == 'node':
                    # Similar to the original paper we only consider a subgraph for explaining
                    feats1 = self.features
                    self.feats1 = feats1
                    graph1 = ptgeom.utils.k_hop_subgraph(n, 3, self.graphs)[1]
                    embeds1 = self.model_to_explain.embedding(feats1, self.graphs).detach()

                    feats2 = self.features
                    self.feats2 = feats2
                    graph2 = ptgeom.utils.k_hop_subgraph(n2, 3, self.graphs)[1]
                    embeds2 = self.model_to_explain.embedding(feats2, self.graphs).detach()

                else:
                    feats1 = self.features[n].detach()
                    graph1 = self.graphs[n].detach()
                    embeds1 = self.model_to_explain.embedding(feats1, graph1).detach()

                    feats2 = self.features[n2].detach()
                    graph2 = self.graphs[n2].detach()
                    embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()

                # Sample possible explanation
                input_expl1, input_expl2 = self._create_explainer_input(graph1, embeds1, n, graph2, embeds2, n2)
                input_expl1 = input_expl1.unsqueeze(0)
                input_expl2 = input_expl2.unsqueeze(0)
                sampling_weights1 = self.explainer_model(input_expl1)
                # sampling_weights1 = torch.mul(sampling_weights1, self.mask1)
                sampling_weights2 = self.explainer_model(input_expl2)
                # sampling_weights2 = torch.mul(sampling_weights2, self.mask2)
                mask1 = self._sample_graph(sampling_weights1, t, bias=self.sample_bias).squeeze()
                mask2 = self._sample_graph(sampling_weights2, t, bias=self.sample_bias).squeeze()
                mask1 = torch.mul(mask1, self.mask1)
                mask2 = torch.mul(mask2, self.mask2)

                t2 = self.dropoutlayer(self.mask2 - mask2)
                mask_pred1 = torch.add(mask1, t2)
                mask_pred1 = torch.sigmoid(mask_pred1)
                t3 = self.dropoutlayer(self.mask1 - mask1)
                mask_pred2 = torch.add(mask2, t3)
                mask_pred2 = torch.sigmoid(mask_pred2)
                masked_pred1, _ = self.model_to_explain(self.merged_feats, self.merged_graph, edge_weights=mask_pred1)
                masked_pred2, _ = self.model_to_explain(self.merged_feats, self.merged_graph, edge_weights=mask_pred2)
                original_pred1, _ = self.model_to_explain(feats1, graph1)
                original_pred2, _ = self.model_to_explain(feats2, graph2)

                if self.type == 'node':  # we only care for the prediction of the node
                    masked_pred1 = masked_pred1[self.index1].unsqueeze(dim=0)
                    original_pred1 = original_pred1[n]

                    masked_pred2 = masked_pred2[self.index2].unsqueeze(dim=0)
                    original_pred2 = original_pred2[n2]

                id_loss = self._loss(masked_pred1, torch.argmax(original_pred1).unsqueeze(0), mask_pred1, self.reg_coefs)
                loss += id_loss
                id_loss = self._loss(masked_pred2, torch.argmax(original_pred2).unsqueeze(0), mask_pred2, self.reg_coefs)
                loss += id_loss
            loss.backward()
            optimizer.step()

    def explain(self, index1):
        """
        Given the index of a node/graph this method returns its explanation. This only gives sensible results if the prepare method has
        already been called.
        :param index: index of the node/graph that we wish to explain
        :return: explanaiton graph and edge weights
        """
        index1 = int(index1)
        index2 = int(self.index2)

        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            graph1 = ptgeom.utils.k_hop_subgraph(index1, 3, self.graphs)[1]
            embeds1 = self.model_to_explain.embedding(self.features, self.graphs).detach()

            graph2 = ptgeom.utils.k_hop_subgraph(index2, 3, self.graphs)[1]
            embeds2 = self.model_to_explain.embedding(self.features, self.graphs).detach()

        else:
            feats1 = self.features[index1].clone().detach()
            graph1 = self.graphs[index1].clone().detach()
            embeds1 = self.model_to_explain.embedding(feats1, graph1).detach()

            feats2 = self.features[index2].detach()
            graph2 = self.graphs[index2].detach()
            embeds2 = self.model_to_explain.embedding(feats2, graph2).detach()

        # Use explainer mlp to get an explanation
        input_expl, _ = self._create_explainer_input(graph1, embeds1, index1, graph2, embeds2, index2)
        input_expl = input_expl.unsqueeze(dim=0)
        sampling_weights = self.explainer_model(input_expl)
        mask = self._sample_graph(sampling_weights, training=False).squeeze()
        final_mask = []
        for i in range(len(self.mask1)):
            # print(self.mask1[i])
            if self.mask1[i] == 1:
                final_mask.append(mask[i])
        expl_graph_weights = torch.zeros(graph1.size(1))  # Combine with original graph
        for i in range(0, len(final_mask)):
            pair = graph1.T[i]
            t = index_edge(graph1, pair)
            expl_graph_weights[t] = final_mask[i]

        return graph1, expl_graph_weights

