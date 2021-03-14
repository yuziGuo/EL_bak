import sqlite3 as lite
import os
import chardet
import json
import sqlite3 as lite
from util_kb import get_cand_info_by_mention, get_text_info_for_entity
from basic_ops_sqlite import test_select_by_cursor
from util_tb import get_table_content, is_measure_col
from constants import base_dir, cache_dir, lite_db_name
from tqdm import tqdm
import threading

test_tb_id = '3389822_6_374624044314151266'
db_name = os.path.join(cache_dir, lite_db_name)


def _create_lite_tb_for_cache(tb_id, logger=None, prefix='tb'):
    # import ipdb; ipdb.set_trace()
    if prefix=='tb':
        schema_str = 'row_id INT, col_id INT, cell_value STRING, lookup_order INT, label STRING, ' \
                     + 'entity_uri STRING, clses STRING, RefCount INT, abstract STRING, comment STRING'
    else:
        schema_str = 'entity_uri STRING, abstract STRING, comment STRING'
    try:
        # import ipdb; ipdb.set_trace()
        con = lite.connect(db_name)
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS '{}_{}'".format(prefix, tb_id))
        cur.execute("CREATE TABLE '{}_{}' ({})".format(prefix, tb_id, schema_str))
    except lite.Error as e:
        if logger:
            logger.error('Error! Message: {}. Fail to create lite tb'.format(e.args[0]))
    return con
    # need to be closed!


def _check_tb_exists(tb_name):
    try:
        con = lite.connect(db_name)
        cur = con.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND tbl_name = '{}';"
                    .format(tb_name))
        rows = cur.fetchall()
    except:
        pass
    return len(rows)==1


def cache_one_table(tb_id, logger=None, support_breakpoint=False):
    # if already cached, skip
    if support_breakpoint and _check_tb_exists('tb_' + tb_id):
        if logger:
            logger.info('Skip! Table {} already cached to {} ! '.format(tb_id, db_name))
        return
    if logger:
        logger.info('Caching table {} to {}'.format(tb_id, db_name))
    con = _create_lite_tb_for_cache(tb_id, logger, prefix='tb')
    col_names = 'row_id, col_id, cell_value, lookup_order, label, ' \
                 + 'entity_uri, clses, RefCount'
    table_cols = get_table_content(tb_id)
    for col_id, col in enumerate(table_cols):  # to +1
        if logger:
            logger.info('Caching column. col_id:{}, row_num:{}'.format(col_id, len(col)))
        if is_measure_col(col):
            if logger:
                logger.info('Judge as measure col! {}'.format(col[:5]))
            continue
        for row_id, cell_item in enumerate(col[1:]):  # to +1
            cell_item = cell_item.strip()
            if cell_item=='':
                continue
            cand_set = get_cand_info_by_mention(row_id+1, col_id, cell_item)
            # import ipdb; ipdb.set_trace()
            if cand_set is None:
                continue
            try:
                cur = con.cursor()
                cur.executemany('INSERT INTO "tb_{}" ({}) VALUES(?,?,?,?, ?,?,?,?)'.format(tb_id, col_names), cand_set)
                con.commit()
            except KeyboardInterrupt:
                con.rollback()  # 会撤销当前这个 cell mention 的 cand set
                sys.exit()
            except lite.Error as e:
                if logger:
                    logger.error('lite.Error {}'.format(e.args[0]))
                con.rollback()
    con.close()

def cache_extending_info_for_one_table(tb_id, logger=None, support_breakpoint=False):
    # if already cached, skip
    if support_breakpoint and _check_tb_exists('extend_by_desc_' + tb_id):
        if logger:
            logger.info('Skip! Table {} already cached to {} ! '.format(tb_id, db_name))
        return
    if logger:
        logger.info('Caching table {}\'s candidates\' info to {}'.format(tb_id, db_name))

    con = _create_lite_tb_for_cache(tb_id, logger, prefix='extend_by_desc')
    col_names = 'entity_uri, abstract, comment'

    entity_uri_list = []
    cur = con.cursor()
    try:
        # import ipdb; ipdb.set_trace()
        cur.execute("SELECT DISTINCT entity_uri FROM 'tb_{}';".format(tb_id))
        for entity in cur.fetchall():
            if type(entity) is tuple:
                entity = entity[0]
            entity_uri_list.append(entity)
    except lite.error as e:
        if logger:
            logger.error("Lite error (caching extending info)! {} tb_id: {}, entity_uri: {}").format(
                e.args[0], tb_id, entity
            )

    res_all = []
    for entity_uri in tqdm(entity_uri_list):
        if logger:  # 1.18s/it 1.13s/it
            logger.info('DBpedia search ! {}'.format(entity_uri))
        res_all.append((entity_uri, )+tuple(get_text_info_for_entity(entity_uri).values()))

    try:
        cur.executemany("INSERT INTO 'extend_by_desc_{}' ({}) VALUES(?,?,?)".format(tb_id, col_names), res_all)
        con.commit()
    except KeyboardInterrupt:
        con.rollback()  # 会撤销当前这个 cell mention 的 cand set
        sys.exit()
    except lite.Error as e:
        if logger:
            logger.error('lite.Error {}'.format(e.args[0]))
        con.rollback()
    con.close()

def _insert_bunch(res_bunch, con, cur, tb_id, col_names):
    try:
        cur.executemany("INSERT INTO 'extend_by_desc_{}' ({}) VALUES(?,?,?)".format(tb_id, col_names),
                        res_bunch)
        con.commit()
        return True
    except KeyboardInterrupt:
        con.rollback()  # 会撤销当前这个 cell mention 的 cand set
        sys.exit()
        return False
    except lite.Error as e:
        if logger:
            logger.error('lite.Error {}'.format(e.args[0]))
        con.rollback()
        return False


def cache_extending_info_for_uri_list(entity_uri_list, flag_list, tb_id, logger=None):
    try:
        con = lite.connect(db_name)
        cur = con.cursor()
    except lite.error as e:
        if logger:
            logger.error("{}: Lite error connecting to db!".threading.currentThread().getName())
    for i in range(3):
        res_all = []
        now_bunch = []
        col_names = 'entity_uri, abstract, comment'
        for entity_uri_idx in tqdm(range(len(entity_uri_list))):
            if flag_list[entity_uri_idx] in (2,1):
                continue
            flag_list[entity_uri_idx] = 1
            now_bunch.append(entity_uri_idx)
            entity_uri = entity_uri_list[entity_uri_idx]
            # if logger:
            #     logger.info('DBpedia search ! {}'.format(entity_uri))
            res_all.append((entity_uri, )+tuple(get_text_info_for_entity(entity_uri).values()))
            if len(res_all)>=100:
                if _insert_bunch(res_all, con, cur, tb_id, col_names) == True:
                    res_all = []
                    for _idx in now_bunch:
                        flag_list[_idx] = 2
                    now_bunch = []
                    logger.info('{}: [YH INFO] 100 inserted !'.format(threading.currentThread().getName()))
                else:
                    res_all = []
                    for _idx in now_bunch:
                        flag_list[_idx] = 0
                    now_bunch = []
                    logger.info('{}: [YH INFO] rollback ! '.format(threading.currentThread().getName()))

        if len(res_all) > 0:
            if _insert_bunch(res_all, con, cur, tb_id, col_names) == True:
                for _idx in now_bunch:
                    flag_list[_idx] = 2
                logger.info('{}: [YH INFO] 100 inserted ! '.format(threading.currentThread().getName()))
            else:
                for _idx in now_bunch:
                    flag_list[_idx] = 0
                logger.info('{}: [YH INFO] rollback ! '.format(threading.currentThread().getName()))
    con.close()


def test_check_tb_exists():
    res1 = _check_tb_exists('tb_1438042986423_95_20150728002306-00125-ip-10-236-191-2_88435628_5')
    res2 = _check_tb_exists('tb_1438042986423_95_20150728002306-00125-ip-10-236-191-2_88435628_000')
    print(res1, res2)


# def test_cache_one_table(tb_id):
#     tb_cache_name = 'tb_'+tb_id
#     test_select_by_cursor(tb_name=tb_cache_name)
#     return



def test_vital(logger):
    # import ipdb; ipdb.set_trace()
    # cache_one_table('1438042989790_89_20150728002309-00310-ip-10-236-191-2_664422904_7')
    # cache_one_table('1438042989891_18_20150728002309-00079-ip-10-236-191-2_64375179_16',logger)
    cache_one_table('62564020_0_3836030043284699244', logger)



if __name__=='__main__':
    # cache_one_table(test_tb_id, logger=None)
    from util_other import getYHLogger
    logger = getYHLogger(prefix='test_cache_extend')
    cache_extending_info_for_one_table(test_tb_id, logger)

    # test_check_tb_exists()

    # from util_other import getYHLogger
    # logger = getYHLogger(prefix='test')
    # test_vital(logger)