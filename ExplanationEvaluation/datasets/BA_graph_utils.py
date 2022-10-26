import random
import numpy as np
import pickle as pkl
# from experiment_models_training.datasets.dataset_loaders import load_dataset
import networkx as nx


def load_ba_shapes():
    graph, preprocessed_features, labels, train_mask, val_mask, test_mask = load_dataset('syn1', skip_preproccessing=True, shuffle=False)

    return []


def build_a_house_graph():

    return


def build_ba_reg(samples=1000, nodes=20, edges=1):
    # rng = random.Random(1234)
    data = []
    for i in range(samples):
        graph = nx.barabasi_albert_graph(n=nodes, m=edges, seed=i)
        # print(graph)
        # print(graph.nodes)
        # print(graph.edges)

        graph.add_nodes_from([20, 21, 22, 23, 24])
        graph.add_edges_from([(20, 21), (21, 22), (22, 23), (23, 24), (24, 20)])
        random.seed(i)
        graph.add_edge(random.randrange(0, 19), random.randrange(20, 24))
        features = []
        for _ in range(25):
            features.append([random.randrange(1, 1000)/10] * 10)
        label = [sum(i[0] * 10 for i in features[20:25])]
        # print(graph.nodes)
        # print(graph.edges)
        # print(features, label)

        # assert 0
        data.append([graph, features, label])
    return data


def build_ba_reg2(samples=1000, nodes=20, edges=1):
    # rng = random.Random(1234)
    data = []
    for i in range(samples):
        random.seed(i)
        num_to_add = random.randrange(1, 20)
        house0 = 120 - 5 * num_to_add
        graph = nx.barabasi_albert_graph(n=120 - 5 * num_to_add, m=edges, seed=i)
        for h in range(num_to_add):
            # graph = nx.barabasi_albert_graph(n=120-5*num_to_add, m=edges, seed=i)
            graph.add_nodes_from([house0, house0 + 1, house0 + 2, house0 + 3, house0 + 4])
            graph.add_edges_from([(house0, house0 + 1), (house0 + 1, house0 + 2), (house0 + 2, house0 + 3),
                                  (house0 + 3, house0 + 4), (house0 + 4, house0)])
            graph.add_edge(random.randrange(0, 120 - 5 * num_to_add - 1), random.randrange(house0, house0 + 4))
            house0 += 5
            # if num_to_add < 20:
            #      graph.add_nodes_from([nid for nid in range(house0, house0 + (20 - num_to_add) * 5)])
        features = []
        num_nodes = 120
        for _ in range(num_nodes):
            features.append([0.1] * 10)
        label = [num_to_add]
        # print(graph.nodes)
        # print(graph.edges)
        # print(features, label)

        # assert 0
        data.append([graph, features, label])
    return data


def save_ba_reg(data, path):
    graphs = []
    features = []
    labels = []
    for i in data:
        features.append(i[1])
        # print(i[1])
        labels.append(i[2])
        # print(i[0])
        # print(i[0].nodes)
        # print(i[0].edges)
        graph = np.zeros((25, 25), dtype=float)
        for edge in i[0].edges:
            # print(i)
            graph[edge[0], edge[1]] = 1.0
            graph[edge[1], edge[0]] = 1.0
        graphs.append(graph)
        # a = abcde
        pass
    graphs = np.asarray(graphs, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    print(features.shape)
    labels = np.asarray(labels, dtype=np.float32)
    # print(graphs[0], features[0], labels[0])
    with open(path, 'wb') as f:
        pkl.dump((graphs, features, labels), f)
    pass


def save_ba_reg2(data, path):
    graphs = []
    features = []
    labels = []
    for i in data:
        tmp = i[1]
        # if len(i[1]) < 120:
        #     tmp += [[0.0] * 10] * (120 - len(i[1]))
        print(len(tmp))
        features.append(tmp)
        labels.append(i[2])
        # print(i[0])
        # print(i[0].nodes)
        # print(i[0].edges)
        graph = np.zeros((120, 120), dtype=float)
        for edge in i[0].edges:
            # print(i)
            graph[edge[0], edge[1]] = 1.0
            graph[edge[1], edge[0]] = 1.0
        graphs.append(graph)
        # a = abcde
        pass
    graphs = np.asarray(graphs, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    # print(graphs[0], features[0], labels[0])
    with open(path, 'wb') as f:
        pkl.dump((graphs, features, labels), f)
    pass


if __name__ == '__main__':
    # data = build_ba_reg()
    # save_ba_reg(data, 'ExplanationEvaluation/datasets/pkls/BA-Reg3.pkl')
    data = build_ba_reg2()
    save_ba_reg2(data, 'ExplanationEvaluation/datasets/pkls/BA-Reg2_2.pkl')
    pass
