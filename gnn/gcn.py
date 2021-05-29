from torch_geometric.data import Data
from py2neo import Graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np

# 读取所有的关系
graph = Graph("http://localhost:7687", auth=("neo4j", "qwer"))


def create_edge(node_dic):
    cypher = 'MATCH p=(n:company)-[r]->(m:company) RETURN id(n),id(m)'
    res = graph.run(cypher)
    df_res = res.to_data_frame()

    node1_id = list(map(lambda x: node_dic[x], df_res['id(n)'].values))
    node2_id = list(map(lambda x: node_dic[x], df_res['id(m)'].values))

    # 构建边
    edge_index = torch.tensor([node1_id, node2_id], dtype=torch.long)

    return edge_index


def create_node():
    cypher = 'MATCH (n) return id(n), properties(n), labels(n)'
    res = graph.run(cypher)
    df_res = res.to_data_frame()

    df_res['labels(n)'].values.tolist()

    return df_res['id(n)'].values.tolist(), process_properties(df_res['properties(n)'])


def process_properties(data):
    industry = []
    assign = []
    violation = []
    bond = []
    dishonesty = []
    for row in data:
        industry.append(row.get('industry', np.nan))
        assign.append(row.get('assign', np.nan))
        violation.append(row.get('violation', np.nan))
        bond.append(row.get('bond', np.nan))
        dishonesty.append(row.get('dishonesty', np.nan))
    df = pd.DataFrame({'industry': industry,
                       'assign': assign,
                       'violation': violation,
                       'bond': bond,
                       'dishonesty': dishonesty})
    # 缺省值填充
    df['industry'].fillna('未知', inplace=True)
    df['assign'].fillna('未知', inplace=True)
    df['violation'].fillna('未知', inplace=True)
    df['bond'].fillna('未知', inplace=True)

    y = df['dishonesty']
    x = pd.get_dummies(df[['industry', 'assign', 'violation', 'bond']])

    return x, y


def create_label():
    cypher = 'MATCH (n:company) return id(n), n.dishonesty'
    res = graph.run(cypher)
    df_res = res.to_data_frame()
    return df_res


def create_graph_data():
    node, (x, y) = create_node()
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
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(35, 64, cached=True)
        self.conv2 = GCNConv(64, 2, cached=True)

        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cpu')
model, data = Net().to(device), dataset.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    logits = model()
    loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
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
