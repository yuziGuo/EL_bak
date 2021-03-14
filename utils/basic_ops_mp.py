import sys
sys.path.append('/home/gyh/pack_for_debug_2/EL_yh/utils/')
from util_cache import cache_extending_info_for_one_table, cache_extending_info_for_uri_list, _create_lite_tb_for_cache
from constants import base_dir, cache_dir, lite_db_name

from threading import Thread
import sqlite3 as lite
import os

def print_args(*args):
    print(args)

def _get_uri_list_by_tb(tb_id):
    lite_tb_name = "{}_{}".format('tb', tb_id)
    db_name = os.path.join(cache_dir, lite_db_name)
    con = lite.connect(db_name)
    cur = con.cursor()
    uri_list = cur.execute("select distinct entity_uri from '{}'".format(lite_tb_name))
    uri_list = list(map(lambda x: x[0] if type(x) == tuple else x, uri_list))
    return uri_list


def test_1():
    from util_other import getYHLogger
    logger = getYHLogger(prefix='test_cache_extend')
    test_tb_id = '3389822_6_374624044314151266'

    for i in range(3):
        t = Thread(target=cache_extending_info_for_one_table, args=(test_tb_id, logger, False))# t = Thread(target=cache_extending_info_for_one_table, args=(test_tb_id, logger, 'test_mp',))
        # t = Thread(target=print_args, args=(test_tb_id, logger, 'test_mp',))
        t.start()
    # while True:
    #     pass


# def cache_extending_info_for_uri_list_with_flags(uri_list, logger):


# 1     16:46:53 - 16:50:24
# 3     15:53:16 - 15:54:27
# 10    16:09:25 - 16:09:47
# 100   16:14:30 - 16:14:31

def test_2():
    from util_other import getYHLogger
    logger = getYHLogger(prefix='test_cache_extend')
    test_tb_id = '3389822_6_374624044314151266'
    uri_list = _get_uri_list_by_tb(test_tb_id)
    flag_list = [False] * len(uri_list)
    # to_lite_name = ''
    tb_id = 'overall'

    _create_lite_tb_for_cache(tb_id, logger, prefix='extend_by_desc')

    for i in range(100):
        t = Thread(target=cache_extending_info_for_uri_list, args=(uri_list, flag_list, tb_id, logger))# t = Thread(target=cache_extending_info_for_one_table, args=(test_tb_id, logger, 'test_mp',))
        # t = Thread(target=print_args, args=(test_tb_id, logger, 'test_mp',))
        t.start()
    # while True:
    #     pass


if __name__=='__main__':
    test_2()
