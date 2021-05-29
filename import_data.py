from py2neo import Node, Subgraph, Graph, Relationship, NodeMatcher
from tqdm import tqdm
import pandas as pd
import numpy as np

graph = Graph("http://127.0.0.1:7474", auth=("neo4j", "qwer"))


def import_company():
    df = pd.read_csv('company_data/公司.csv')
    eid = df['eid'].values
    name = df['companyname'].values

    nodes = []
    data = list(zip(eid, name))
    for eid, name in tqdm(data):
        profit = np.random.randint(100000, 100000000, 1)[0]
        node = Node('company', name=name, profit=int(profit), eid=eid)
        nodes.append(node)

    graph.create(Subgraph(nodes))


def import_person():
    df = pd.read_csv('company_data/人物.csv')
    pid = df['personcode'].values
    name = df['personname'].values

    nodes = []
    data = list(zip(pid, name))
    for eid, name in tqdm(data):
        age = np.random.randint(20, 70, 1)[0]
        node = Node('person', name=name, age=int(age), pid=str(eid))
        nodes.append(node)

    graph.create(Subgraph(nodes))


def import_industry():
    df = pd.read_csv('company_data/行业.csv')
    names = df['orgtype'].values

    nodes = []
    for name in tqdm(names):
        node = Node('industry', name=name)
        nodes.append(node)

    graph.create(Subgraph(nodes))


def import_assign():
    df = pd.read_csv('company_data/分红.csv')
    names = df['schemetype'].values

    nodes = []
    for name in tqdm(names):
        node = Node('assign', name=name)
        nodes.append(node)

    graph.create(Subgraph(nodes))


def import_violations():
    df = pd.read_csv('company_data/违规类型.csv')
    names = df['gooltype'].values

    nodes = []
    for name in tqdm(names):
        node = Node('violations', name=name)
        nodes.append(node)

    graph.create(Subgraph(nodes))


def import_bond():
    df = pd.read_csv('company_data/债券类型.csv')
    names = df['securitytype'].values

    nodes = []
    for name in tqdm(names):
        node = Node('bond', name=name)
        nodes.append(node)

    graph.create(Subgraph(nodes))


# def import_dishonesty():
#     node = Node('dishonesty', name='失信')
#     graph.create(node)


def import_relation():
    df = pd.read_csv('company_data/公司-人物.csv')
    matcher = NodeMatcher(graph)
    eid = df['eid'].values
    pid = df['pid'].values
    post = df['post'].values
    relations = []
    data = list(zip(eid, pid, post))
    for e, p, po in tqdm(data):
        company = matcher.match('company', eid=e).first()
        person = matcher.match('person', pid=str(p)).first()
        if company is not None and person is not None:
            relations.append(Relationship(company, po, person))

    graph.create(Subgraph(relationships=relations))
    print('import company-person relation succeeded')

    df = pd.read_csv('company_data/公司-行业.csv')
    matcher = NodeMatcher(graph)
    eid = df['eid'].values
    name = df['industry'].values
    relations = []
    data = list(zip(eid, name))
    for e, n in tqdm(data):
        company = matcher.match('company', eid=e).first()
        industry = matcher.match('industry', name=str(n)).first()
        if company is not None and industry is not None:
            relations.append(Relationship(company, '行业类型', industry))

    graph.create(Subgraph(relationships=relations))
    print('import company-industry relation succeeded')

    df = pd.read_csv('company_data/公司-分红.csv')
    matcher = NodeMatcher(graph)
    eid = df['eid'].values
    name = df['assign'].values
    relations = []
    data = list(zip(eid, name))
    for e, n in tqdm(data):
        company = matcher.match('company', eid=e).first()
        assign = matcher.match('assign', name=str(n)).first()
        if company is not None and assign is not None:
            relations.append(Relationship(company, '分红方式', assign))

    graph.create(Subgraph(relationships=relations))
    print('import company-assign relation succeeded')

    df = pd.read_csv('company_data/公司-违规.csv')
    matcher = NodeMatcher(graph)
    eid = df['eid'].values
    name = df['violations'].values
    relations = []
    data = list(zip(eid, name))
    for e, n in tqdm(data):
        company = matcher.match('company', eid=e).first()
        violations = matcher.match('violations', name=str(n)).first()
        if company is not None and violations is not None:
            relations.append(Relationship(company, '违规类型', violations))

    graph.create(Subgraph(relationships=relations))
    print('import company-violations relation succeeded')

    df = pd.read_csv('company_data/公司-债券.csv')
    matcher = NodeMatcher(graph)
    eid = df['eid'].values
    name = df['bond'].values
    relations = []
    data = list(zip(eid, name))
    for e, n in tqdm(data):
        company = matcher.match('company', eid=e).first()
        bond = matcher.match('bond', name=str(n)).first()
        if company is not None and bond is not None:
            relations.append(Relationship(company, '债券类型', bond))

    graph.create(Subgraph(relationships=relations))
    print('import company-bond relation succeeded')

    # df = pd.read_csv('company_data/公司-失信.csv')
    # matcher = NodeMatcher(graph)
    # eid = df['eid'].values
    # rel = df['dishonesty'].values
    # relations = []
    # data = list(zip(eid, rel))
    # for e, r in tqdm(data):
    #     company = matcher.match('company', eid=e).first()
    #     dishonesty = matcher.match('dishonesty', name='失信').first()
    #     if company is not None and dishonesty is not None:
    #         if pd.notna(r):
    #             if int(r) == 0:
    #                 relations.append(Relationship(company, '无', dishonesty))
    #             elif int(r) == 1:
    #                 relations.append(Relationship(company, '有', dishonesty))
    #
    # graph.create(Subgraph(relationships=relations))
    # print('import company-dishonesty relation succeeded')


def import_company_relation():
    df = pd.read_csv('company_data/公司-供应商.csv')
    matcher = NodeMatcher(graph)
    eid1 = df['eid1'].values
    eid2 = df['eid2'].values
    relations = []
    data = list(zip(eid1, eid2))
    for e1, e2 in tqdm(data):
        if pd.notna(e1) and pd.notna(e2) and e1 != e2:
            company1 = matcher.match('company', eid=e1).first()
            company2 = matcher.match('company', eid=e2).first()

            if company1 is not None and company2 is not None:
                relations.append(Relationship(company1, '供应商', company2))

    graph.create(Subgraph(relationships=relations))
    print('import company-supplier relation succeeded')

    df = pd.read_csv('company_data/公司-担保.csv')
    matcher = NodeMatcher(graph)
    eid1 = df['eid1'].values
    eid2 = df['eid2'].values
    relations = []
    data = list(zip(eid1, eid2))
    for e1, e2 in tqdm(data):
        if pd.notna(e1) and pd.notna(e2) and e1 != e2:
            company1 = matcher.match('company', eid=e1).first()
            company2 = matcher.match('company', eid=e2).first()

            if company1 is not None and company2 is not None:
                relations.append(Relationship(company1, '担保', company2))

    graph.create(Subgraph(relationships=relations))
    print('import company-guarantee relation succeeded')

    df = pd.read_csv('company_data/公司-客户.csv')
    matcher = NodeMatcher(graph)
    eid1 = df['eid1'].values
    eid2 = df['eid2'].values
    relations = []
    data = list(zip(eid1, eid2))
    for e1, e2 in tqdm(data):
        if pd.notna(e1) and pd.notna(e2):
            company1 = matcher.match('company', eid=e1).first()
            company2 = matcher.match('company', eid=e2).first()

            if company1 is not None and company2 is not None:
                relations.append(Relationship(company1, '客户', company2))

    graph.create(Subgraph(relationships=relations))
    print('import company-customer relation succeeded')


def delete_relation():
    cypher = 'match ()-[r]-() delete r'
    graph.run(cypher)


def delete_node():
    cypher = 'match (n) delete n'
    graph.run(cypher)


def import_data():
    import_company()
    import_company_relation()

    import_person()
    import_industry()
    import_assign()
    import_violations()
    import_bond()
    # import_dishonesty()

    import_relation()


def delete_data():
    delete_relation()
    delete_node()
    print('delete data succeeded')


if __name__ == '__main__':
    profit = np.random.randint(100000, 100000000, 10).tolist()

    delete_data()
    import_data()