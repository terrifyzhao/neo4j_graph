from torch_geometric.data import Data
from py2neo import Graph
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import SAGEConv

# 读取所有的关系
graph = Graph("http://localhost:7474", auth=("neo4j", "qwer"))


def create_edge(node_dic):
    cypher = 'MATCH p=(n:company)-[r]->(m:company) RETURN n.name,m.name'
    res = graph.run(cypher)
    df_res = res.to_data_frame()

    node1_id = list(map(lambda x: node_dic[x], df_res['n.name'].values))
    node2_id = list(map(lambda x: node_dic[x], df_res['m.name'].values))

    # 构建边
    edge_index = torch.tensor([node1_id, node2_id], dtype=torch.long)

    return edge_index


def create_node():
    cypher = 'MATCH (n) return n.name, labels(n)'
    res = graph.run(cypher)
    df_res = res.to_data_frame()

    df_pro = pd.read_csv('../company.csv')
    df = pd.concat([df_res, df_pro],
                   axis=1,
                   keys=['n.name', 'companyname'],
                   join='outer')

    y = df['companyname']['dishonesty_y']

    df_x = df['companyname']
    x = pd.get_dummies(df_x[['industry', 'assign', 'violations', 'bond']])

    return df['n.name']['n.name'], x, y


def create_label():
    cypher = 'MATCH (n:company) return id(n), n.dishonesty'
    res = graph.run(cypher)
    df_res = res.to_data_frame()
    return df_res


def create_graph_data():
    node, x, y = create_node()
    node_dic = {}
    for i in node:
        node_dic[i] = len(node_dic)

    edge_index = create_edge(node_dic)

    y = list(map(lambda x: -1 if pd.isna(x) else x, y))
    # 训练集
    train_mask = list(map(lambda x: True if x != -1 else False, y))
    # 测试集，要排除非公司节点
    test_mask = [False] * 2922 + list(map(lambda x: True if x == -1 else False, y[2922:]))
    dataset = Data(torch.tensor(x.values, dtype=torch.float),
                   edge_index=edge_index,
                   y=torch.tensor(y, dtype=torch.long),
                   train_mask=torch.tensor(train_mask),
                   test_mask=torch.tensor(test_mask))
    return dataset


dataset = create_graph_data()

print(dataset)


class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        in_channels = 31
        out_channels = 2
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self):
        x0, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1 = F.relu(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.relu(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.lin(x)
        return x


device = torch.device('cpu')
model, data = Net(64).to(device), dataset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    logits = model()
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    pred = logits[data.train_mask].max(1)[1]
    acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    loss.backward()
    optimizer.step()
    return loss, acc


best_acc = 0
for epoch in range(1, 201):
    loss, acc = train()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pkl')
    log = 'epoch: {:3d}, loss: {:.4f}, train acc: {:.4f}'
    print(log.format(epoch, loss.item(), acc))


# model.load_state_dict(torch.load('best_model.pkl'))

def inference():
    logits = model()
    pred = logits[data.test_mask].max(1)[1]
    return pred


res = inference()
print(res)