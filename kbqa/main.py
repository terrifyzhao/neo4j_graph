import torch
from kbqa.data_process import *
import ahocorasick
from py2neo import Graph

model = torch.load('text_cnn.p')
model.eval()

graph = Graph("http://192.168.50.179:7474", auth=("neo4j", "qwer"))
company = graph.run('MATCH (n:company) RETURN n.name as name').to_ndarray()
person = graph.run('MATCH (n:person) RETURN n.name as name').to_ndarray()
relation = graph.run('MATCH ()-[r]-() RETURN distinct type(r)').to_ndarray()

ac_company = ahocorasick.Automaton()
ac_person = ahocorasick.Automaton()
ac_relation = ahocorasick.Automaton()

for key in enumerate(company):
    ac_company.add_word(key[1][0], key[1][0])
for key in enumerate(person):
    ac_person.add_word(key[1][0], key[1][0])
for key in enumerate(relation):
    ac_relation.add_word(key[1][0], key[1][0])

ac_company.make_automaton()
ac_person.make_automaton()
ac_relation.make_automaton()


def classification_predict(s):
    s = seq2index(s)
    s = torch.from_numpy(padding_seq([s])).cuda().long()
    out = model(s)
    out = out.cpu().data.numpy()
    print(out)
    return out.argmax(1)[0]


def entity_link(text):
    subject = []
    subject_type = None
    for end_index, original_value in ac_company.iter(text):
        start_index = end_index - len(original_value) + 1
        print('实体：', (start_index, end_index, original_value))
        assert text[start_index:start_index + len(original_value)] == original_value
        subject.append(original_value)
        subject_type = 'company'
    for end_index, original_value in ac_person.iter(text):
        start_index = end_index - len(original_value) + 1
        print('实体：', (start_index, end_index, original_value))
        assert text[start_index:start_index + len(original_value)] == original_value
        subject.append(original_value)
        subject_type = 'person'

    return subject[0], subject_type


def kbqa(text):
    cls = classification_predict(text)
    print('question type:', cls)
    cypher = ''
    res = ''
    if cls == 0:
        pass
    elif cls == 1:
        # 查询属性
        subject, subject_type = entity_link(text)
        predicate = ''
        for w in ['年龄', '年纪']:
            if w in text and subject_type == 'person':
                predicate = 'age'
                break
        for w in ['收入', '收益']:
            if w in text and subject_type == 'company':
                predicate = 'profit'
                break
        cypher = f'''match (n:{subject_type}) where n.name='{subject}' return n.{predicate}'''
        print(cypher)
        res = graph.run(cypher).to_ndarray()
    elif cls == 2:
        pass

    return res


while 1:
    text = input('text:')
    res = kbqa(text)
    print(res)
