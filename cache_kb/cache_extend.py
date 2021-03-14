import sys
sys.path.append('/home/gyh/pack_for_debug_2/EL_yh/utils')
from constants import cache_dir, lite_db_name, base_dir
from util_other import getYHLogger
from util_cache import cache_extending_info_for_one_table, \
    cache_extending_info_for_uri_list, \
    _create_lite_tb_for_cache, \
    _check_tb_exists
from threading import Thread
import os
db_name = os.path.join(cache_dir, lite_db_name)
import sqlite3 as lite


def _get_uri_list_overall():
    tb_list = os.listdir(os.path.join(base_dir, 'instance'))
    tb_list = list(map(lambda x: '.'.join(x.split('.')[:-1]), tb_list))

    con = lite.connect(db_name)
    cur = con.cursor()

    entity_set = set()
    for tb_name in tb_list:
        cur.execute("SELECT DISTINCT entity_uri FROM 'tb_{}'".format(tb_name))
        for u in cur.fetchall():
            u = u[0] if type(u) == tuple else u
            entity_set.add(u)
    con.close()
    return list(entity_set)

def _get_uri_list_cached(tb_id):
    lite_tb_name = 'extend_by_desc_'+tb_id
    con = lite.connect(db_name)
    cur = con.execute("SELECT DISTINCT entity_uri FROM '{}'".format(lite_tb_name))
    ent_list = cur.fetchall()
    ent_list = list(map(lambda x: x[0] if type(x)==tuple else x, ent_list))
    return ent_list


def cache(support_breakpoint=False):
    import ipdb; ipdb.set_trace()
    logger = getYHLogger(prefix='cache_extend_overall')
    tb_id = 'overall'

    uri_list = _get_uri_list_overall()
    if support_breakpoint:
        cached_list = _get_uri_list_cached(tb_id)
        # uri_list = list(filter(lambda x: x not in cached_list, uri_list))
        uri_list = list(set(uri_list).difference(set(cached_list)))
    flag_list = [0] * len(uri_list)
    logger.info('{} entity uris to search and cache'.format(len(uri_list)))

    if not support_breakpoint or not _check_tb_exists('extend_by_desc_'+tb_id):
        _create_lite_tb_for_cache(tb_id, logger, prefix='extend_by_desc')

    thread_num = 30
    for i in range(thread_num):
        t = Thread(target=cache_extending_info_for_uri_list, args=(uri_list, flag_list, tb_id, logger))
        t.start()

if __name__=='__main__':
    cache(support_breakpoint=True)