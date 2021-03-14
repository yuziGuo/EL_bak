import os
import chardet
import json
# import nltk
import re
base_dir = '/home/gyh/pack_for_debug_2/EL_yh/data/t2d'

'''
About webtable:

dict_keys(['relation', 'pageTitle', 'title', 'url', 'hasHeader', 
'headerPosition', 'tableType', 'tableNum', 's3Link', 'recordEndOffset', 
'recordOffset', 'tableOrientation', 'lastModified', 'textBeforeTable', 
'textAfterTable', 'hasKeyColumn', 'keyColumnIndex', 'headerRowIndex'])
["pageTitle", "title", "textBeforeTable", "textAfterTable"]

'''


def webtable2dict(tb_id):
    line = ''
    # tb_id = tb_id.split('.')[0]
    if tb_id.endswith('csv') or tb_id.endswith('json'):
        tb_id = '.'.join(tb_id.split('.')[:-1])
    fn = '.'.join([tb_id, 'json'])
    with open(os.path.join(base_dir, 'tables', fn), 'rb') as f:
        line_b = f.readlines()[0]
        chardet_info = chardet.detect(line_b)
        try:
            assert chardet_info['confidence'] > 0.65
            line += line_b.decode(chardet_info['encoding'])
        except:
            AssertionError
            print('Unrecognized encoding schema', fn)
        try:
            t = json.loads(line)
        except json.JSONDecodeError as e:
            print('Error: {} '.format(args[0]))
            logger.error('Json error when decoding table {}'.format(fn))
        # finally:
        #     table_cols = t['relation']
        return t


def get_table_content(tb_id):
    return webtable2dict(tb_id)['relation']


def get_table_desc(tb_id, desc_keys=["pageTitle", "title", "textBeforeTable", "textAfterTable"]):
    # return webtable2dict(tb_id)['title']
    webtable = webtable2dict(tb_id)
    return {k: webtable.get(k, None) for k in desc_keys}


def is_numeric(num):
    try:
        float(num)
    except:
        return False
    return True

# def is_digit_col_easy(col):
#     try:
#         assert len(col) >= 2
#     except AssertionError:
#         print({'Error! Empty table without cell contents.'})
#         return
#     col = col[1:]
#     return is_numeric(col[0]) and is_numeric(col[-1])

#
# def is_digit_col(col):
#     try:
#         assert len(col) >= 2
#     except AssertionError:
#         print({'Error! Empty table without cell contents.'})
#         return
#     col = col[1:]
#     col = list(filter(lambda x: x.strip()!='', col))
#     # case 1
#     short_col = [col[0], col[-1]]
#     if is_numeric(col[0]) and is_numeric(col[-1]):
#         return True
#     # case 2
#     short_col_splited = list(map(lambda x: nltk.regexp_tokenize(x, r'(?u)\d+(?:\.\d+)?|\w+'), short_col))
#     for _ in short_col_splited:
#         try:
#             if not is_numeric(_[0]):
#                 return False
#         except IndexError as e:
#             import ipdb; ipdb.set_trace()
#
#
#     # case 3
#     # short table -> judge it as not numeric;
#     # 1. not enough rows to find repeting pattern in one column;
#     # 2. acceptable cost to search kb
#     if len(col) < 20:
#         return False
#
#     # case 4
#     col = col[:20]
#     col_splited = list(map(lambda x: nltk.regexp_tokenize(x, r'(?u)\d+(?:\.\d+)?|\w+')[1:], col))
#     # col splited: [['km', 'h'], ['km', 'h'], ...]
#     # ['km'] is unhashable! so we cannot use set(col_splited)
#     try:
#         col_splited = list(map(lambda x: '_'.join(x), col_splited))
#         if len(set(col_splited)) < 6:
#             return True
#     except:
#         import ipdb; ipdb.set_trace()
#
#     # finally
#     return False

# def is_quantity_col(col):
#     try:
#         assert len(col) >= 2
#     except AssertionError:
#         print({'Error! Empty table without cell contents.'})
#         return
#     col = col[1:]
#     # import ipdb; ipdb.set_trace()
#     if len(col) < 5:
#         return False
#     if len(col) > 10:
#         return (len(set(''.join(list(set(col))))) < 8 or len(set(col)) < 5) and max([len(_) for _ in col]) <= 5
#     return (len(set(list(set(col)))) < 6 or len(set(col)) < 4) and max([len(_) for _ in col]) < 5
#

# def test_1():
#     # is_digit_col(['height', '183.0'])
#     # res = is_digit_col(['fake_col_name', '183.0 km/h', '11.3 sec/h'])
#     import ipdb; ipdb.set_trace()
#     res = is_digit_col(['6,446,000', '1,170,000', '294,000', '584,000', '1,382,000', '512,000', '1,089,000', '506,000', '494,000', '864,000', '670,000', '2,810,000', '680,000', ''])
#     print(res)

def test_2():
    # content = get_table_content('3389822_6_374624044314151266')
    desc = get_table_desc('3389822_6_374624044314151266')
    print(desc)


def is_measure_col(col):
    try:
        assert len(col) >= 2
    except AssertionError:
        print({'Error! Empty table without cell contents.'})
        return
    col = col[1:]
    col = list(filter(lambda x: x.strip()!='', col))
    
    col_splited = [re.match(r'^([\d+(,\.\d+)]*)[\s]*([a-zA-Z\/]*)', cell).groups() for cell in col]
    # import ipdb; ipdb.set_trace()
    # col_numeric_part = [_[0] for _ in col_splited]
    col_measure_part = [_[1] for _ in col_splited]
    measures = set()
    for _ in col_measure_part:
        for _ in re.findall(r'([\d+(,\.\d+)]+|[a-z]+)', _):
            measures.add(_)
    if len(measures) <= min(len(col)//5, 7) and \
            len(set(''.join(list(measures))))<20:
        # 1. a few possible words (km h ...)
        # magic number 7: 1('') + 3 + 3
        # 2. a few possible chars (to exclude ['China', 'US', ...])
        return True
    return False

def test_3():
    cols = get_table_content('3389822_6_374624044314151266')
    for col in cols:
        print(col[:10])
        # print(is_quantity_col(col))
        print(is_measure_col(col))


def test_4():
    # re.match(r'^([\d+(,\.\d+)]*)[\s]*([a-zA-Z\/]*)', '1,111,111.km/h').groups()
    col_0 = [
        '111', '0',
        '111.1','1,111,111.0',
        '1,111 km/h', '11. %',
    ]

    col_1 = [
        'GS', 'GS/MS', 'MS', 'GK', 'GS', 'GS/MS', 'MS', 'GK'
    ]

    col_2 = ['6,446,000', '1,170,000', '294,000', '584,000', '1,382,000',
             '512,000', '1,089,000', '506,000', '494,000', '864,000', '670,000',
             '2,810,000', '680,000', '']


    cols = [col_0, col_1, col_2]
    for _ in cols:
        print(_)
        print(is_measure_col(_))


if __name__=='__main__':
    test_3()
    test_4()
