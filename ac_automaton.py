import ahocorasick
from py2neo import Graph

graph = Graph("http://localhost:7474", auth=("neo4j", "qwer"))
company = graph.run('MATCH (n:company) RETURN n.name as name').to_ndarray()
relation = graph.run('MATCH ()-[r]-() RETURN distinct type(r)').to_ndarray()

ac_company = ahocorasick.Automaton()
ac_relation = ahocorasick.Automaton()

for key in enumerate(company):
    ac_company.add_word(key[1][0], key[1][0])
for key in enumerate(relation):
    ac_relation.add_word(key[1][0], key[1][0])

ac_company.make_automaton()
ac_relation.make_automaton()

# haystack = '浙江东阳东欣房地产开发有限公司的客户的供应商'
haystack = '衡水中南锦衡房地产有限公司的债券类型'
# haystack = '临沂金丰公社农业服务有限公司的分红方式'
print('question:', haystack)

subject = ''
predicate = []

for end_index, original_value in ac_company.iter(haystack):
    start_index = end_index - len(original_value) + 1
    print('公司实体：', (start_index, end_index, original_value))
    assert haystack[start_index:start_index + len(original_value)] == original_value
    subject = original_value

for end_index, original_value in ac_relation.iter(haystack):
    start_index = end_index - len(original_value) + 1
    print('关系：', (start_index, end_index, original_value))
    assert haystack[start_index:start_index + len(original_value)] == original_value
    predicate.append(original_value)

for p in predicate:
    cypher = f'''match (s:company)-[p:`{p}`]-(o) where s.name='{subject}' return o.name'''
    print(cypher)
    res = graph.run(cypher).to_ndarray()
    # print(res)
    subject = res[0][0]
print('answer:', res[0][0])