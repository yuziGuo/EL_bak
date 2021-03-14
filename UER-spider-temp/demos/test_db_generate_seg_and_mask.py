from col_spec_yh.encode_utils_db import generate_seg_with_meta
from col_spec_yh.encode_utils_db import *
from col_spec_yh.store_utils import decode_sqlite_spider_tbls_in_rows_with_metadata

from demos.utils import get_args_minimal as get_args
from demos.samples.sample_mini_tables import table_a, table_b, table_with_empty_values_1, table_with_empty_values_2

def test_generate_seg_with_meta_1():
    args = get_args()
    args.seq_len = 128
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=True)
    tokens_1, seg_1 = generate_seg_with_meta(args, table_b, row_wise_fill=True)

    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    args.high_level_clses = ['TAB', 'COL']
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    args.high_level_clses = ['TAB', 'CELL']
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    args.high_level_clses = ['TAB']
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')
    import ipdb; ipdb.set_trace()

def test_generate_seg_with_meta_1():
    args = get_args()
    args.seq_len = 128
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=True)
    tokens_1, seg_1 = generate_seg_with_meta(args, table_b, row_wise_fill=True)

    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    args.high_level_clses = ['TAB', 'COL']
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    args.high_level_clses = ['TAB', 'CELL']
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    args.high_level_clses = ['TAB']
    tokens_0, seg_0 = generate_seg_with_meta(args, table_a, row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')
    import ipdb; ipdb.set_trace()


def test_generate_seg_with_meta_2():
    args = get_args()
    args.seq_len = 128
    args.high_level_clses = ['TAB', 'COL', 'CELL', 'NL']
    tokens_0, seg_0 = generate_seg_with_meta(args, [], row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    tokens_0, seg_0 = generate_seg_with_meta(args, [], nl=['I', 'love', 'you'], row_wise_fill=False)
    check_segs(zip([seg_0], [tokens_0]), option='col')

    import ipdb; ipdb.set_trace()

def _test_high_cls_option():
    args = get_args()
    args.seq_len = 128

    print('args.has_high_level: False; row-wise')
    args.has_high_level_cls = False
    tokens_0, seg_0 = generate_seg(args, table_a, row_wise_fill=True)
    tokens_1, seg_1 = generate_seg(args, table_b, row_wise_fill=True)
    seg = torch.LongTensor([seg_0, seg_1])
    check_segs(zip([seg_0, seg_1], [tokens_0, tokens_1]))
    import ipdb; ipdb.set_trace()

    print('args.has_high_level: False; col-wise')
    args.has_high_level_cls = False
    tokens_0, seg_0 = generate_seg(args, table_a, row_wise_fill=False)
    tokens_1, seg_1 = generate_seg(args, table_b, row_wise_fill=False)
    seg = torch.LongTensor([seg_0, seg_1])
    check_segs(zip([seg_0, seg_1], [tokens_0, tokens_1]))
    import ipdb; ipdb.set_trace()

    print('args.has_high_level: True; row-wise')
    args.has_high_level_cls = True
    tokens_0, seg_0 = generate_seg(args, table_a, row_wise_fill=True)
    tokens_1, seg_1 = generate_seg(args, table_b, row_wise_fill=True)
    seg = torch.LongTensor([seg_0, seg_1])
    check_segs(zip([seg_0, seg_1], [tokens_0, tokens_1]))
    import ipdb; ipdb.set_trace()

    print('args.has_high_level: True; col-wise')
    args.has_high_level_cls = True
    tokens_0, seg_0 = generate_seg(args, table_a, row_wise_fill=False)
    tokens_1, seg_1 = generate_seg(args, table_b, row_wise_fill=False)
    seg = torch.LongTensor([seg_0, seg_1])
    check_segs(zip([seg_0, seg_1], [tokens_0, tokens_1]))
    import ipdb; ipdb.set_trace()



def check_segs(iter, option='cell'):
    args = get_args()
    _cell_idx = lambda x : x % 10000
    _row_idx = lambda x : x % 10000 % 100
    _col_idx = lambda x : x % 10000 // 100
    for (seg, tokens) in iter:
        if option == 'cell':
            f = _cell_idx
        if option == 'row':
            f = _row_idx
        if option == 'col':
            f = _col_idx
        i = 0; s = 0; now = f(seg[s])
        while s < len(seg):
            while i < len(seg) and f(seg[i]) == now: i += 1
            print(seg[s:i])
            print(args.tokenizer.convert_ids_to_tokens(tokens[s:i]))
            s = i
            if s < len(seg):
                now = f(seg[s])


from col_spec_yh.store_utils import decode_sqlite_spider_tbls_in_rows_with_metadata
from col_spec_yh.encode_utils_db import *
def test_1():
    db_path = './data/spider/slim_99/farm/farm.sqlite'
    _ = './data/spider/processed_db.json'
    _ = decode_sqlite_spider_tbls_in_rows_with_metadata(db_path, _)
    args = get_args()
    for tid, _ in _.items():
        rows = _['rows']
        cols = list(zip(*rows))
        tokens, seg = generate_seg_with_meta(args, cols, tbl_id=None, title=_['title'],header=_['header'], dgl_backend=True)
        # node_start_id = ?
        max_node_id = len(seg)-1
        seg = torch.LongTensor(seg)
        import ipdb; ipdb.set_trace()
        meta_idx = f_is_meta(seg)

        print(seg)
        print(tokens)
        check_segs([(seg, tokens)], 'row')

if __name__=='__main__':
    # test_1()
    # test_generate_seg_with_meta_1()
    test_generate_seg_with_meta_2()