import json
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
        edge_mask = torch.ones(edge_index.size(1))
        data = Data(x=x, edge_index=edge_index, edge_mask=edge_mask, y=y)
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

    if eval_enabled:
        model.eval()

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
        out, _ = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max)
        optimizer.step()

        if args.eval_enabled: model.eval()
        with torch.no_grad():
            out, _ = model(x, edge_index)

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
    out, _ = model(x, edge_index)

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

    if_gpu = 0
    if torch.cuda.is_available():
        print('cuda is available!')
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())
        if_gpu = 1
        n_gpu = torch.cuda.device_count()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(_paper)
    graphs, features, labels, train_mask, val_mask, test_mask = load_dataset(_dataset, _paper)
    graphs_alt_1, features_alt_1, labels_alt_1, _, _, _ = load_dataset(_dataset + '-alt-1', _paper)
    graphs_alt_2, features_alt_2, labels_alt_2, _, _, _ = load_dataset(_dataset + '-alt-2', _paper)
    graphs_alt_3, features_alt_3, labels_alt_3, _, _, _ = load_dataset(_dataset + '-alt-3', _paper)
    graphs_alt_4, features_alt_4, labels_alt_4, _, _, _ = load_dataset(_dataset + '-alt-4', _paper)
    graphs_alt_5, features_alt_5, labels_alt_5, _, _, _ = load_dataset(_dataset + '-alt-5', _paper)
    graphs_alt_6, features_alt_6, labels_alt_6, _, _, _ = load_dataset(_dataset + '-alt-6', _paper)
    graphs_alt_7, features_alt_7, labels_alt_7, _, _, _ = load_dataset(_dataset + '-alt-7', _paper)
    graphs_alt_8, features_alt_8, labels_alt_8, _, _, _ = load_dataset(_dataset + '-alt-8', _paper)
    train_set = create_data_list(graphs, features, labels, train_mask)
    val_set = create_data_list(graphs, features, labels, val_mask)
    test_set = create_data_list(graphs, features, labels, test_mask)

    val_set_alt_1 = create_data_list(graphs_alt_1, features_alt_1, labels_alt_1, val_mask)
    val_set_alt_2 = create_data_list(graphs_alt_2, features_alt_2, labels_alt_2, val_mask)
    val_set_alt_3 = create_data_list(graphs_alt_3, features_alt_3, labels_alt_3, val_mask)
    val_set_alt_4 = create_data_list(graphs_alt_4, features_alt_4, labels_alt_4, val_mask)
    val_set_alt_5 = create_data_list(graphs_alt_5, features_alt_5, labels_alt_5, val_mask)
    val_set_alt_6 = create_data_list(graphs_alt_6, features_alt_6, labels_alt_6, val_mask)
    val_set_alt_7 = create_data_list(graphs_alt_7, features_alt_7, labels_alt_7, val_mask)
    val_set_alt_8 = create_data_list(graphs_alt_8, features_alt_8, labels_alt_8, val_mask)

    print('len dataset: ', len(train_set), len(val_set), len(test_set), len(val_set_alt_1), len(val_set_alt_2),
          len(val_set_alt_3), len(val_set_alt_4))
    print('len dataset: ', len(val_set_alt_5), len(val_set_alt_6),
          len(val_set_alt_7), len(val_set_alt_8))
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
        val_alt_5_loader = DataLoader(val_set_alt_5, batch_size=len(val_set), shuffle=False)
        val_alt_6_loader = DataLoader(val_set_alt_6, batch_size=len(val_set), shuffle=False)
        val_alt_7_loader = DataLoader(val_set_alt_7, batch_size=len(val_set), shuffle=False)
        val_alt_8_loader = DataLoader(val_set_alt_8, batch_size=len(val_set), shuffle=False)
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    # Define graph
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_rmse = 999999999
    best_val_mape = 99999
    best_epoch = 0

    only_do_eval = True
    if only_do_eval:
        model = load_best_model(2849, _paper, _dataset, model, True).to(device)

        def val_embedding(val_alt_loader, val_set_alt):
            ebds = []
            for data in val_alt_loader:
                data.to(device)
                _, ebd = model(data.x, data.edge_index, data.edge_mask, data.batch)
                ebds.append(ebd)
            return ebds

        val_ebd = val_embedding(val_loader, val_set)
        val_ebd_1 = val_embedding(val_alt_1_loader, val_set_alt_1)
        val_ebd_2 = val_embedding(val_alt_2_loader, val_set_alt_2)
        val_ebd_3 = val_embedding(val_alt_3_loader, val_set_alt_3)
        val_ebd_4 = val_embedding(val_alt_4_loader, val_set_alt_4)
        val_ebd_5 = val_embedding(val_alt_5_loader, val_set_alt_5)
        val_ebd_6 = val_embedding(val_alt_6_loader, val_set_alt_6)
        val_ebd_7 = val_embedding(val_alt_7_loader, val_set_alt_7)
        val_ebd_8 = val_embedding(val_alt_8_loader, val_set_alt_8)
        print(len(val_ebd))
        print(val_ebd[0].shape)
        # print(val_ebd[0].detach().tolist())

        val_ebd = val_ebd[0].detach().tolist()
        val_ebd_1 = val_ebd_1[0].detach().tolist()
        val_ebd_2 = val_ebd_2[0].detach().tolist()
        val_ebd_3 = val_ebd_3[0].detach().tolist()
        val_ebd_4 = val_ebd_4[0].detach().tolist()
        val_ebd_5 = val_ebd_5[0].detach().tolist()
        val_ebd_6 = val_ebd_6[0].detach().tolist()
        val_ebd_7 = val_ebd_7[0].detach().tolist()
        val_ebd_8 = val_ebd_8[0].detach().tolist()

        res = [val_ebd, val_ebd_1, val_ebd_2, val_ebd_3, val_ebd_4, val_ebd_5, val_ebd_6, val_ebd_7, val_ebd_8]
        with open('./results/val_embedding.json', 'w') as f:
            f.write(json.dumps(res))
        return

    for epoch in range(0, args.epochs):
        model.train()

        # Use pytorch-geometric batching method
        for data in train_loader:
            data.to(device)
            optimizer.zero_grad()
            out, _ = model(data.x, data.edge_index, data.edge_mask, data.batch)
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
                data.to(device)
                out, _ = model(data.x, data.edge_index, data.edge_mask, data.batch)
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
            out, _ = model(eval_data.x, eval_data.edge_index, eval_data.batch)
            test_rmse = math.sqrt(model.loss(out, eval_data.y))

            def do_val_alt(val_alt_loader, val_set_alt):
                val_loss, val_sum, val_mape = 0.0, 0.0, 0.0
                for data in val_alt_loader:
                    data.to(device)
                    out, _ = model(data.x, data.edge_index, data.edge_mask, data.batch)
                    val_loss += model.loss(out, data.y)
                    val_sum += model.loss(out, data.y)
                    val_mape += model.mape(out, data.y)
                # print('val mape: ', val_mape, len(val_set_alt))
                val_rmse = math.sqrt(val_sum / len(val_set_alt))
                val_mape = float(val_mape) / int(len(val_set_alt))
                return val_rmse, val_mape, val_loss

            val_rmse, val_mape, val_loss = do_val_alt(val_loader, val_set)

        # print(f"Epoch: {epoch}, train_rmse: {train_rmse:.4f}, val_rmse: {val_rmse:.4f}, train_loss: {train_loss:.4f}")

        if val_mape < best_val_mape:  # New best results
            val_rmse_1, val_mape_1, _ = do_val_alt(val_alt_1_loader, val_set_alt_1)
            val_rmse_2, val_mape_2, _ = do_val_alt(val_alt_2_loader, val_set_alt_2)
            val_rmse_3, val_mape_3, _ = do_val_alt(val_alt_3_loader, val_set_alt_3)
            val_rmse_4, val_mape_4, _ = do_val_alt(val_alt_4_loader, val_set_alt_4)
            val_rmse_5, val_mape_5, _ = do_val_alt(val_alt_5_loader, val_set_alt_5)
            val_rmse_6, val_mape_6, _ = do_val_alt(val_alt_6_loader, val_set_alt_6)
            val_rmse_7, val_mape_7, _ = do_val_alt(val_alt_7_loader, val_set_alt_7)
            val_rmse_8, val_mape_8, _ = do_val_alt(val_alt_8_loader, val_set_alt_8)
            print("Val improved")
            print(f"Epoch: {epoch}, train_rmse: {train_rmse:.6f}, train_mape: {train_mape: .6f}, val_rmse: {val_rmse:.6f}, val_mape: {val_mape: .6f},  train_loss: {train_loss:.4f}")
            print(
                f"val_rmse_alt1: {val_rmse_1:.6f}, val_mape_alt1: {val_mape_1: .6f}, val_rmse_alt2: {val_rmse_2:.6f}, val_mape_alt2: {val_mape_2: .6f}, ")
            print(
                f"val_rmse_alt3: {val_rmse_3:.6f}, val_mape_alt3: {val_mape_3: .6f}, val_rmse_alt4: {val_rmse_4:.6f}, val_mape_alt4: {val_mape_4: .6f}, ")
            print(
                f"val_rmse_alt5: {val_rmse_5:.6f}, val_mape_alt5: {val_mape_5: .6f}, val_rmse_alt6: {val_rmse_6:.6f}, val_mape_alt6: {val_mape_6: .6f}, ")
            print(
                f"val_rmse_alt7: {val_rmse_7:.6f}, val_mape_alt7: {val_mape_7: .6f}, val_rmse_alt8: {val_rmse_8:.6f}, val_mape_alt8: {val_mape_8: .6f}, ")

            best_val_rmse = val_rmse
            best_val_mape = val_mape
            best_epoch = epoch
            store_checkpoint(_paper, _dataset, model, train_rmse, val_rmse, test_rmse, best_epoch)

        # Early stopping
        if epoch - best_epoch > args.early_stopping:
            break
