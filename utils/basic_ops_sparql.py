import time
import sparql
endpoint = 'http://dbpedia.org/sparql'


def test_single_query():
    handle = sparql.Service('http://dbpedia.org/sparql', "utf-8")
    statement = \
        'PREFIX foaf:  <http://xmlns.com/foaf/0.1/> ' \
        'PREFIX dbo:   <http://dbpedia.org/ontology/> ' \
        'PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#> ' \
        'PREFIX dbr:   <http://dbpedia.org/resource/> ' \
        \
        'SELECT ?abs, ?comment ' \
        'WHERE {' \
        'dbr:Apple_II dbo:abstract ?abs . ' \
        'dbr:Apple_II rdfs:comment ?comment . ' \
        'FILTER (langMatches(lang(?abs),"en")) ' \
        'FILTER (langMatches(lang(?comment),"en")) ' \
        '} ' \
        'limit 10'
    t = time.time()
    for _ in range(100):
        result = handle.query(statement)
        print(time.time() - t)
        t = time.time()
        res_all = result.fetchall()
        keys = result.variables
        values = res_all[0] if len(res_all) > 0 else ['Nothing']*len(keys)
        r = {k:values[idx].value[:10] for idx, k in enumerate(keys)}
        print(r)
    return


def _test_connection_1():
    statement = 'select distinct ?Concept where {[] a ?Concept} LIMIT 100'
    res = sparql.query(endpoint, statement)
    print(res.fetchall())


def _test_connection_2():
    s = sparql.Service(endpoint, "utf-8")
    statement = ('select distinct ?Concept where {[] a ?Concept} LIMIT 100')
    res = s.query(statement)
    print(res.fetchall())


def test_1():
    _test_connection_1()
    _test_connection_2()


if __name__ == '__main__':
    test_single_query()

