import sys
sys.path.append('/home/gyh/pack_for_debug_2/EL_yh/utils')
from constants import base_dir
from util_cache import cache_one_table, _create_lite_tb_for_cache
from util_other import getYHLogger
import os
from tqdm import tqdm
import sqlite3 as lite


def cache_all():
    d = os.path.join(base_dir, 'instance')
    test_table_list = list(map(lambda x: '.'.join(x.split('.')[:-1]), os.listdir(d)))

    global_cache_logger = getYHLogger(prefix='cache_step_1')
    global_cache_logger.info('Caching from {}', test_table_list)

    for _ in tqdm(test_table_list):
        cache_one_table(_, global_cache_logger, support_breakpoint=True)


def _clean(tb_id):
    try:
        from constants import cache_dir, lite_db_name
        db_name = os.path.join(cache_dir, lite_db_name)
        con = lite.connect(db_name)
        cur = con.cursor()
        cur.execute('DELETE FROM "tb_{}" WHERE cell_value=""'.format(tb_id))
    except lite.Error as e:
        con.rollback()
    return


def _clean_all():
    d = os.path.join(base_dir, 'instance')
    test_table_list = list(map(lambda x: '.'.join(x.split('.')[:-1]), os.listdir(d)))
    for _ in tqdm(test_table_list):
        _clean(_)

# fix
def _fix():
    tb_id = '1438042989043_35_20150728002309-00201-ip-10-236-191-2_448144'
    con = _create_lite_tb_for_cache(tb_id, logger=None, prefix='tb_' )
    con.close()


if __name__=='__main__':
    cache_all()
    # _clean_all()
    # _fix()

