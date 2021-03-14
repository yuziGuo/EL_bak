# from util_kb import lookup_entities
import requests
import sparql
import xml.etree.ElementTree as ET
from urllib.error import HTTPError


def _look_up_mention(cell_item='fly me to the moon', top_k=50):
    lookup_url = 'http://localhost:9274/lookup-application/api/search?query=%s&MaxHits=%d' \
                 % (cell_item, top_k)
    lookup_res = requests.get(lookup_url)
    if lookup_res.status_code>400:  # 400, 500
        return []
    root = ET.fromstring(lookup_res.content)
    cand_set = []
    for candidate_idx, child in enumerate(root):
        res = {}
        res['mention'] = cell_item
        res['lookup_order'] = candidate_idx
        res['label'] = child[0].text
        res['uri'] = child[1].text # should add format checking
        res['clses'] = []
        for cls in child[2]:
            res['clses'].append(cls[0].text)
        res['clses'] = str(res['clses'])
        res['refCnt'] = child[3].text
        res['refCnt'] = int(res['refCnt']) if res['refCnt'] else None
        cand_set.append(res)
    return cand_set



'''
1. 基本用法
2. 长问句 http://localhost:9274/lookup-application/api/search?query=fly%20me%20to%20the%20moon&MaxHits=10
'''
def get_cand_info_by_mention(row_id, col_id, cell_item='fly me to the moon', top_k=50):
    if len(cell_item) > 50:
        return None
    res_by_dict = _look_up_mention(cell_item, top_k)
    if res_by_dict == []:
        return None
    tem = [(row_id, col_id,)+tuple(_.values()) for _ in res_by_dict]
    # ['row_id', 'col_id']+['mention', 'lookup_order', 'label', 'uri', 'clses', 'refCnt']
    return tuple(tem)


def _gen_sparql_for_abstract_and_comment(entity_uri):
    statement = '''
        PREFIX foaf:  <http://xmlns.com/foaf/0.1/> 
        PREFIX dbo:   <http://dbpedia.org/ontology/> 
        PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#> 
        PREFIX dbr:   <http://dbpedia.org/resource/>

        SELECT ?abstract, ?comment 
        WHERE {
            <%s> dbo:abstract ?abstract . 
            <%s> rdfs:comment ?comment . 
            FILTER (langMatches(lang(?abstract),"en")) 
            FILTER (langMatches(lang(?comment),"en")) 
        } 
        limit 10
    ''' % (entity_uri, entity_uri)
    return statement


def get_text_info_for_entity(entity_uri):
    handle = sparql.Service('http://dbpedia.org/sparql', "utf-8")
    statement = _gen_sparql_for_abstract_and_comment(entity_uri)
    try:
        result = handle.query(statement)
    # except HTTPError as e:
    except:
        import ipdb; ipdb.set_trace()
        print(e.args[0])
    res_all = result.fetchall()
    keys = result.variables
    values = ['Nothing'] * len(keys)
    try:
        if len(res_all) > 0:
            _v = res_all[0]
            for idx in range(len(values)):
                values[idx] = _v[idx].value if _v[idx] is not None else 'Nothing'
    except:
        import ipdb; ipdb.set_trace()
    r = {k: values[idx] for idx, k in enumerate(keys)}
    # print(r)
    return r


def _test_api_connection(cell_item='fly me to the moon'):
    cand_set = _look_up_mention(cell_item)
    print(cand_set)


def test_1():
    test_api_connection()
    pass


def test_2():
    entity_uri = 'http://dbpedia.org/resource/Piz_Urlaun'
    info = get_text_info_for_entity(entity_uri)
    print(info)


if __name__=='__main__':
    # test_2()
    # print(gen_sparql('http://dbpedia.org/resource/Apple_II'))
    en = "http://dbpedia.org/resource/Davison's_Mill,_Stelling_Minnis"
    print(_gen_sparql_for_abstract_and_comment(en))
    pass



