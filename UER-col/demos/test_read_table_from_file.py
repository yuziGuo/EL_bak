from col_spec_yh.store_utils import decode_and_verify_aida_file
from col_spec_yh.store_utils import decode_and_verify_spider_file
from col_spec_yh.store_utils import get_labels_map_from_aida_file_2
from col_spec_yh.store_utils import decode_and_verify_aida_file_2
from col_spec_yh.store_utils import decode_aida_file_2_one_table
from col_spec_yh.store_utils import decode_and_verify_aida_file_2


def test_1():
    tab_file = 'demos/samples/sample_file_type0-1.tb'
    tab_cols_list = decode_and_verify_spider_file(tab_file)
    print(tab_cols_list)


def test_2():
    tab_file = 'data/aida/ff_no_dup_train_samples'
    tab_cols_list = decode_and_verify_aida_file(tab_file)
    import ipdb; ipdb.set_trace()
    print(tab_cols_list)


def test_3():
    path = '/home/gyh/pack_for_debug_2/EL_yh/UER-col/col_cls_workspace/IO/train_samples'
    labels_map = get_labels_map_from_aida_file_2(path)
    print(labels_map)


def test_4():
    path = '/home/gyh/pack_for_debug_2/EL_yh/UER-col/col_cls_workspace/IO/train_samples/Writer-6869358_0_1379459120563510331 3'
    micro_tables = decode_aida_file_2_one_table(path)
    print(micro_tables)


def test_5():
    data_path = '/home/gyh/pack_for_debug_2/EL_yh/UER-col/col_cls_workspace/IO/train_samples'
    _ = decode_and_verify_aida_file_2(data_path)


if __name__=='__main__':
    test_2()
