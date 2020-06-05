from torch_geometric.data import Data
from py2neo import Graph
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# 读取所有的关系
graph = Graph("http://localhost:11004", auth=("neo4j", "qwer"))


def create_edge(node_dic):
    cypher = 'MATCH p=(n:company)-[r]->(m:company) RETURN id(n),id(m)'
    res = graph.run(cypher)
    df_res = res.to_data_frame()

    node1_id = list(map(lambda x: node_dic[x], df_res['id(n)'].values))
    node2_id = list(map(lambda x: node_dic[x], df_res['id(m)'].values))

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
    node = create_node()
    node_dic = {}
    for i in node:
        node_dic[i] = len(node_dic)

    node_dic, edge_index = create_edge(node_dic)
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
    train_mask = list(map(lambda x: True if x != -1 else False, y))
    test_mask = list(map(lambda x: True if x == -1 else False, y))
    dataset = Data(torch.tensor([[1] for _ in range(len(node))], dtype=torch.float),
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


for epoch in range(1, 201):
    loss, acc = train()
    log = 'epoch: {:3d}, loss: {:.4f}, train acc: {:.4f}'
    print(log.format(epoch, loss.item(), acc))


def inference():
    logits = model()
    pred = logits[data.test_mask].max(1)[1]
    print(data.y[data.test_mask])
    return pred


res = inference()
print(res)
