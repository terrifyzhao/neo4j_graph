from py2neo import Node, Subgraph, Graph, Relationship, NodeMatcher
from tqdm import tqdm
import pandas as pd

graph = Graph("http://localhost:7474", auth=("neo4j", "qwer"))


def import_company():
    df = pd.read_csv('company_data/公司.csv')
    eid = df['eid'].values
    name = df['companyname'].values

    nodes = []
    for eid, name in tqdm(zip(eid, name)):
        node = Node('company', name=name, eid=eid)
        nodes.append(node)

    graph.create(Subgraph(nodes))


def import_person():
    df = pd.read_csv('company_data/人物2.csv')
    pid = df['personcode'].values
    name = df['personname'].values

    nodes = []
    for eid, name in tqdm(zip(pid, name)):
        node = Node('person', name=name, pid=str(eid))
        nodes.append(node)

    graph.create(Subgraph(nodes))


def import_relationship():
    df = pd.read_csv('company_data/公司-人物2.csv')
    matcher = NodeMatcher(graph)
    eid = df['eid'].values
    pid = df['pid'].values
    post = df['post'].values
    relations = []
    for e, p, po in tqdm(zip(eid, pid, post)):
        company = matcher.match('company', eid=e).first()
        person = matcher.match('person', pid=str(p)).first()
        if company is not None and person is not None:
            relations.append(Relationship(company, po, person))

    graph.create(Subgraph(relationships=relations))


def delete_relation():
    cypher = 'match ()-[r]-() delete r'
    graph.run(cypher)


def delete_node():
    cypher = 'match (n) delete n'
    graph.run(cypher)


def import_data():
    import_company()
    import_person()
    import_relationship()


def delete_data():
    delete_relation()
    delete_node()


if __name__ == '__main__':
    delete_data()
    import_data()
