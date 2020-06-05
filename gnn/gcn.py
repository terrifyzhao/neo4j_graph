from torch_geometric.data import Data
from py2neo import Graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 读取所有的关系
graph = Graph("http://localhost:11004", auth=("neo4j", "qwer"))


def create_edge():
    cypher = 'MATCH p=(n:company)-[r]->(m:company) RETURN id(n),id(m)'
    res = graph.run(cypher)
    df_res = res.to_data_frame()

    # 所有节点
    node1_id = df_res['id(n)'].values.tolist()
    node2_id = df_res['id(m)'].values.tolist()
    node_id = []
    node_id.extend(node1_id)
    node_id.extend(node2_id)

    # 节点重新命名id
    node_id = list(set(node_id))
    node_dic = {}
    for i in node_id:
        node_dic[i] = len(node_dic)
    node1_id = list(map(lambda x: node_dic[x], node1_id))
    node2_id = list(map(lambda x: node_dic[x], node2_id))

    # 构建边
    edge_index = torch.tensor([node1_id, node2_id], dtype=torch.long)

    return node_dic, edge_index


def create_node():
    cypher = 'MATCH (n:company) return id(n)'
    res = graph.run(cypher)
    df_res = res.to_data_frame()
    return df_res['id(n)'].values.tolist()


def create_label():
    cypher = 'MATCH p=(n)-[r:有|无]->(m) RETURN id(n),type(r)'
    res = graph.run(cypher)
    df_res = res.to_data_frame()
    return df_res


def create_graph_data():
    node_dic, edge_index = create_edge()
    node = create_node()
    node = list(map(lambda x: node_dic[x], node))
    label_df = create_label()
    label_df['id(n)'] = list(map(lambda x: node_dic[x], label_df['id(n)'].values))
    label_df = label_df.sample(frac=1)
    label_dic = {}
    for k, v in zip(label_df['id(n)'], label_df['type(r)']):
        if v == '有':
            v = 1
        else:
            v = 0
        label_dic[k] = v

    y = list(map(lambda x: label_dic[x] if x in label_dic.keys() else -1, node))
    train_mask = list(map(lambda x: True if x is not None else False, y))
    test_mask = list(map(lambda x: True if x is None else False, y))
    dataset = Data(torch.tensor([[1] * len(node)], dtype=torch.float),
                   edge_index=edge_index,
                   y=torch.tensor(y, dtype=torch.long),
                   train_mask=torch.tensor(train_mask),
                   test_mask=torch.tensor(test_mask))
    return dataset


dataset = create_graph_data()

print(dataset)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 16, cached=True)
        self.conv2 = GCNConv(16, 2, cached=True)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cpu')

model, data = Net().to(device), dataset.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    # loss = F.nll_loss(model(data), data.y)
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


# @torch.no_grad()
# def test():
#     model.eval()
#     logits = model()
#     pred = logits.max(1)[1]
#     acc = pred.eq(data.y).sum().item() / len(df_res)
#     return acc


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    loss = train()
    acc = test()
    log = 'epoch: {:3d}, loss: {:.4f}, acc: {:.4f}'
    print(log.format(epoch, loss.item(), acc))
    # train_acc, val_acc, tmp_test_acc = test()
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     test_acc = tmp_test_acc
    # log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, best_val_acc, test_acc))
