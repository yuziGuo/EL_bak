import os

from col_spec_yh.constants import *
from collections import defaultdict


def decode_and_verify_spider_file(data_path):
    tab_cols_list = []
    with open(data_path) as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n') #
            _cols = line.split(ROW_SEP)
            _cols = [_.split(COL_SEP) for _ in _cols]
            # ipdb.set_trace()
            try:
                assert [len(_) for _ in _cols] == [len(_cols[0])] * len(_cols)
                # print(_cols)
                tab_cols_list.append(_cols)
            except:
                AssertionError
                ipdb.set_trace()
                print(_cols)
                ipdb.set_trace()
    print('Checked! ' + data_path)
    return tab_cols_list


def decode_and_verify_aida_file(data_path):
    tab_cols_list = []
    label_list = []
    raw_tab_id_list = []
    with open(data_path) as f:
        for idx, csv_line in enumerate(f):
            csv_line = csv_line.rstrip('\n') #
            [raw_tab_info, cls_name, _main_col, _sur_cols] = csv_line.split(CSV_SEP)
            raw_tab_id_list.append(':'.join(raw_tab_info.split(':')[:-1]))
            label_list.append(cls_name)
            _main_col = _main_col.split(COL_SEP)
            _sur_cols = _sur_cols.split(ROW_SEP)
            _sur_cols = [_.split(COL_SEP) for _ in _sur_cols]
            _cols = [_main_col] + _sur_cols
            # ipdb.set_trace()
            try:
                assert [len(_) for _ in _cols] == [len(_cols[0])] * len(_cols)
                # print(_cols)
                tab_cols_list.append(_cols)
            except:
                AssertionError
                ipdb.set_trace()
                print(_cols)
                ipdb.set_trace()
    print('Checked! DataPath: {}. Lines (#micro tables): {}'.format(data_path, len(label_list)))
    return raw_tab_id_list, label_list, tab_cols_list


def decode_rows_from_aida_file_2_one_table(data_path):
    lines = []
    with open(data_path) as f:
        lines.extend(f.readlines())
    rows = [row.rstrip('\n').split(ROW_SEP) for row in lines]
    return rows


# def decode_aida_file_2_one_table(data_path):
#     # data_path = '/home/gyh/pack_for_debug_2/EL_yh/UER-col/col_cls_workspace/IO/train_samples/Writer-6869358_0_1379459120563510331 3'
#     rows = decode_rows_from_aida_file_2_one_table(data_path)
#     random.shuffle(rows)
#     micro_tables = [list(zip(*rows[i:i+5])) for i in range(0, len(rows), 5)]
#     return micro_tables




def decode_and_verify_aida_file_2(data_path=None):
    tab_cols_list = []
    label_list = []
    raw_tab_id_list = []

    f_name_list = os.listdir(data_path)

    for f_name in f_name_list:
        label_name = f_name.split('-')[0]
        raw_tab_id = '-'.join(f_name.split('-')[1:])
        micro_tables = decode_aida_file_2_one_table(os.path.join(data_path, f_name))
        tab_cols_list.extend(micro_tables)
        label_list.extend([label_name] * len(micro_tables))
        raw_tab_id_list.extend([raw_tab_id] * len(micro_tables))
    # if shuffle:
    #     _t = list(zip(raw_tab_id_list, label_list, tab_cols_list))
    #     random.shuffle(_t)
    #     return list(zip(*_t))
    return raw_tab_id_list, label_list, tab_cols_list


def get_labels_map_from_aida_file(data_path):
    cls_names = []
    with open(data_path) as f:
        for idx, csv_line in enumerate(f):
            csv_line = csv_line.rstrip('\n')
            cls_name = csv_line.split(CSV_SEP)[1]
            cls_names.append(cls_name)
    labels_map = {k: i for i, k in enumerate(list(set(cls_names)))}
    return labels_map

#
def get_labels_map_from_aida_file_2(data_path):
    cls_names = [_.split('-')[0] for _ in  os.listdir(data_path)]
    cls_names = list(set(cls_names))
    cls_names.sort()
    # ['a', 'b'] --> {'a': 0, 'b': 1}
    labels_map = {v: k for k, v in enumerate(cls_names)}
    return labels_map


def decode_aida_ds_in_rows(data_path=None):
    tb_to_rows = defaultdict(list)
    tb_to_cls_name = defaultdict(str)
    f_name_list = os.listdir(data_path)
    for f_name in f_name_list:
        label_name = f_name.split('-')[0]
        raw_tab_id = '-'.join(f_name.split('-')[1:])
        tb_to_cls_name[raw_tab_id] = label_name
        tb_to_rows[raw_tab_id] = decode_rows_from_aida_file_2_one_table(os.path.join(data_path, f_name))
    return tb_to_cls_name, tb_to_rows


def decode_sqlite_spider_tbls_in_rows_with_metadata(db_path, preprocess_f_path):
    '''
    :param db_path:
    :param preprocess_f_path:
    :return:
    '''
    import sqlite3 as lite
    import json
    import os
    from collections import defaultdict


    def f(): return {'title': '', 'header': [], 'rows': []}
    ret = defaultdict(f)

    con = lite.connect(db_path)
    cur = con.cursor()
    db = os.path.basename(db_path).split('.')[0]
    _ = json.load(open(preprocess_f_path))[db]
    for idx, (tbl_ori, tbl_lemma) in enumerate(zip(_['table_names_original'], _['table_names_lemma'])):
        if tbl_ori == 'sqlite_sequence':
            tbl_ori = 'fake_yh_sqlite_sequence'
            continue
        try:
            rows = cur.execute('SELECT * FROM {}'.format(tbl_ori)).fetchall()
        except:
            import ipdb; ipdb.set_trace()
            print(db)
        ret[idx]['title'] = tbl_lemma
        ret[idx]['rows'] = rows
        ret[idx]['header'] = [_[1] for _ in _['column_names_lemma'] if _[0]==idx]
    con.close()
    return ret

def test():
    db_path = './data/spider/slim_99/farm/farm.sqlite'
    _ = './data/spider/processed_db.json'
    _ = decode_sqlite_spider_tbls_in_rows_with_metadata(db_path, _)
    import ipdb; ipdb.set_trace()

    for tbl in _:
        print(tbl['table_name'])
        print(tbl['header'])
        print(tbl['rows'])


if __name__=='__main__':
    test()