import math
import os
import torch
import numpy as np
from torch_geometric.data import Data, DataLoader

from dataset_loaders import load_dataset
from model_selector import model_selector


def create_data_list(graphs, features, labels, mask):
    """
    Convert the numpy data to torch tensors and save them in a list.
    :params graphs: edge indecs of the graphs
    :params features: features for every node
    :params labels: ground truth labels
    :params mask: mask, used to filter the data
    :retuns: list; contains the dataset
    """
    indices = np.argwhere(mask).squeeze()
    data_list = []
    for i in indices:
        x = torch.tensor(features[i])
        edge_index = torch.tensor(graphs[i])
        y = torch.tensor(labels[i])
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
        # print(data)
    return data_list


def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    :param out: predicted outputs of the explainer
    :param labels: ground truth of the data
    :returns: int accuracy
    """
    # preds = out.argmax(dim=1)
    # correct = preds == labels
    # acc = int(correct.sum()) / int(correct.size(0))
    print(out)
    print(labels)
    print(out.shape, labels.shape)
    rmse = torch.nn.MSELoss(out, labels)
    rmse /= int(out.size(0))
    return rmse


def store_checkpoint(paper, dataset, model, train_acc, val_acc, test_acc, epoch=-1):
    """
    Store the model weights at a predifined location.
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters we whish to save
    :param train_acc: training accuracy obtained by the model
    :param val_acc: validation accuracy obtained by the model
    :param test_acc: test accuracy obtained by the model
    :param epoch: the current epoch of the training process
    :retunrs: None
    """
    save_dir = f"./checkpoints/{paper}/{dataset}"
    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_dir, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_dir, f"model_{epoch}"))


def load_best_model(best_epoch, paper, dataset, model, eval_enabled):
    """
    Load the model parameters from a checkpoint into a model
    :param best_epoch: the epoch which obtained the best result. use -1 to chose the "best model"
    :param paper: str, the paper 
    :param dataset: str, the dataset
    :param model: the model who's parameters overide
    :param eval_enabled: wheater to activate evaluation mode on the model or not
    :return: model with pramaters taken from the checkpoint
    """
    print(best_epoch)
    if best_epoch == -1:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/best_model")
    else:
        checkpoint = torch.load(f"./checkpoints/{paper}/{dataset}/model_{best_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    if eval_enabled: model.eval()

    return model


def train_node(_dataset, _paper, args):
    """
    Train a explainer to explain node classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "PG"
    :param args: a dict containing the relevant model arguements
    """
    graph, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset)
    model = model_selector(_paper, _dataset, False)

    x = torch.tensor(features)
    edge_index = torch.tensor(graph)
    labels = torch.tensor(labels)

    # Define graph
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(0, args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out = model(x, edge_index)

        # Evaluate train
        train_acc = evaluate(out[train_mask], labels[train_mask])
        test_acc = evaluate(out[test_mask], labels[test_mask])
        val_acc = evaluate(out[val_mask], labels[val_mask])

        print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")

        if val_acc > best_val_acc:  # New best results
            print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc, best_epoch)

        if epoch - best_epoch > args.early_stopping and best_val_acc > 0.99:
            break

    model = load_best_model(best_epoch, _paper, _dataset, model, args.eval_enabled)
    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[train_mask], labels[train_mask])
    test_acc = evaluate(out[test_mask], labels[test_mask])
    val_acc = evaluate(out[val_mask], labels[val_mask])
    print(f"final train_acc:{train_acc}, val_acc: {val_acc}, test_acc: {test_acc}")

    store_checkpoint(_paper, _dataset, model, train_acc, val_acc, test_acc)


def train_graph(_dataset, _paper, args):
    """
    Train a explainer to explain graph classifications
    :param _dataset: the dataset we wish to use for training
    :param _paper: the paper we whish to follow, chose from "GNN" or "REG"
    :param args: a dict containing the relevant model arguements
    """
    print(_paper)
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset, _paper)
    graphs_alt_1, features_alt_1, labels_alt_1, _, _, _ = load_dataset(_dataset + '-alt-1', _paper)
    graphs_alt_2, features_alt_2, labels_alt_2, _, _, _ = load_dataset(_dataset + '-alt-2', _paper)
    graphs_alt_3, features_alt_3, labels_alt_3, _, _, _ = load_dataset(_dataset + '-alt-3', _paper)
    graphs_alt_4, features_alt_4, labels_alt_4, _, _, _ = load_dataset(_dataset + '-alt-4', _paper)
    train_set = create_data_list(graphs, features, labels, train_mask)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)

    val_set_alt_1 = create_data_list(graphs_alt_1, features_alt_1, labels_alt_1, val_mask)
    val_set_alt_2 = create_data_list(graphs_alt_2, features_alt_2, labels_alt_2, val_mask)
    val_set_alt_3 = create_data_list(graphs_alt_3, features_alt_3, labels_alt_3, val_mask)
    val_set_alt_4 = create_data_list(graphs_alt_4, features_alt_4, labels_alt_4, val_mask)

    print('len dataset: ', len(train_set), len(val_set), len(test_set), len(val_set_alt_1), len(val_set_alt_2),
          len(val_set_alt_3), len(val_set_alt_4))
    model = model_selector(_paper, _dataset, False)

    print(type(model))
    if _paper == 'REG':
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
        val_alt_1_loader = DataLoader(val_set_alt_1, batch_size=len(val_set), shuffle=False)
        val_alt_2_loader = DataLoader(val_set_alt_2, batch_size=len(val_set), shuffle=False)
        val_alt_3_loader = DataLoader(val_set_alt_3, batch_size=len(val_set), shuffle=False)
        val_alt_4_loader = DataLoader(val_set_alt_4, batch_size=len(val_set), shuffle=False)
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_rmse = 999999999
    best_val_mape = 99999
    best_epoch = 0

    for epoch in range(0, args.epochs):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            # print(out)
            # print(out.shape)
            # print(data.y)
            loss = model.mape(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
            optimizer.step()
            # assert 0
        model.eval()
        # Evaluate train
        with torch.no_grad():
            train_sum = 0
            loss = 0
            train_mape = 0
            for data in train_loader:
                out = model(data.x, data.edge_index, data.batch)
                # print(out, data.y)
                loss += model.mape(out, data.y)
                # print(criterion(out, data.y))
                # preds = out.argmax(dim=1)
                train_sum += model.loss(out, data.y)
                train_mape += model.mape(out, data.y)
                # assert 0
            train_rmse = math.sqrt(float(train_sum) / int(len(train_set)))
            train_mape = float(train_mape) / int(len(train_set))
            train_loss = float(loss) / int(len(train_set))

            eval_data = next(iter(test_loader))  # Loads all test samples
            out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            test_rmse = math.sqrt(model.loss(out, eval_data.y))

            val_loss, val_sum, val_mape = 0.0, 0.0, 0.0
            for data in val_loader:
                out = model(data.x, data.edge_index, data.batch)
                # print(out, data.y)
                val_loss += model.loss(out, data.y)
                # print(criterion(out, data.y))
                # preds = out.argmax(dim=1)
                val_sum += model.loss(out, data.y)
                val_mape += model.mape(out, data.y)
            # eval_data = next(iter(val_loader))  # Loads all eval samples
            # out = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            print('val mape: ', val_mape, len(val_set))
            val_rmse = math.sqrt(val_sum/len(val_set))
            val_mape = float(val_mape) / int(len(val_set))
            # print(train_mape, val_mape)

            val_loss_1, val_sum_1, val_mape_1 = 0.0, 0.0, 0.0
            for data in val_alt_1_loader:
                out = model(data.x, data.edge_index, data.batch)
                val_loss_1 += model.loss(out, data.y)
                val_sum_1 += model.loss(out, data.y)
                val_mape_1 += model.mape(out, data.y)
            print('val mape_1: ', val_mape_1, len(val_set_alt_1))
            val_rmse_1 = math.sqrt(val_sum_1 / len(val_set_alt_1))
            val_mape_1 = float(val_mape_1) / int(len(val_set_alt_1))

            val_loss_2, val_sum_2, val_mape_2 = 0.0, 0.0, 0.0
            for data in val_alt_2_loader:
                out = model(data.x, data.edge_index, data.batch)
                val_loss_2 += model.loss(out, data.y)
                val_sum_2 += model.loss(out, data.y)
                val_mape_2 += model.mape(out, data.y)
            print('val mape_2: ', val_mape_2, len(val_set_alt_2))
            val_rmse_2 = math.sqrt(val_sum_2 / len(val_set_alt_2))
            val_mape_2 = float(val_mape_2) / int(len(val_set_alt_2))

            val_loss_3, val_sum_3, val_mape_3 = 0.0, 0.0, 0.0
            for data in val_alt_3_loader:
                out = model(data.x, data.edge_index, data.batch)
                val_loss_3 += model.loss(out, data.y)
                val_sum_3 += model.loss(out, data.y)
                val_mape_3 += model.mape(out, data.y)
            print('val mape_3: ', val_mape_3, len(val_set_alt_3))
            val_rmse_3 = math.sqrt(val_sum_3 / len(val_set_alt_3))
            val_mape_3 = float(val_mape_3) / int(len(val_set_alt_3))

            val_loss_4, val_sum_4, val_mape_4 = 0.0, 0.0, 0.0
            for data in val_alt_4_loader:
                out = model(data.x, data.edge_index, data.batch)
                val_loss_4 += model.loss(out, data.y)
                val_sum_4 += model.loss(out, data.y)
                val_mape_4 += model.mape(out, data.y)
            print('val mape_4: ', val_mape_4, len(val_set_alt_4))
            val_rmse_4 = math.sqrt(val_sum_4 / len(val_set_alt_4))
            val_mape_4 = float(val_mape_4) / int(len(val_set_alt_4))
        # print(f"Epoch: {epoch}, train_rmse: {train_rmse:.4f}, val_rmse: {val_rmse:.4f}, train_loss: {train_loss:.4f}")

        if val_rmse < best_val_rmse or val_mape < best_val_mape:  # New best results
            print("Val improved")
            print(f"Epoch: {epoch}, train_rmse: {train_rmse:.6f}, train_mape: {train_mape: .6f}, val_rmse: {val_rmse:.6f}, val_mape: {val_mape: .6f},  train_loss: {train_loss:.4f}")
            print(
                f"val_rmse_alt1: {val_rmse_1:.6f}, val_mape_alt1: {val_mape_1: .6f}, val_rmse_alt2: {val_rmse_2:.6f}, val_mape_alt2: {val_mape_2: .6f}, ")
            print(
                f"val_rmse_alt3: {val_rmse_3:.6f}, val_mape_alt3: {val_mape_3: .6f}, val_rmse_alt4: {val_rmse_4:.6f}, val_mape_alt4: {val_mape_4: .6f}, ")

            best_val_rmse = val_rmse
            best_val_mape = val_mape
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_rmse, val_rmse, test_rmse, best_epoch)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break
