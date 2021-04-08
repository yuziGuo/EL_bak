import torch
from col_spec_yh.constants import *
from col_spec_yh.store_utils import decode_sqlite_spider_tbls_in_rows_with_metadata
from demos.utils import get_args_minimal as get_args
import random

_tbl_idx = lambda x: x % 100

_row_idx = lambda x: x % 100
_col_idx = lambda x: x % 10000 // 100
_cell_idx = lambda x: x % 10000

# Use: */& instead of and, +/| instead of or, ~ instead of not
_is_tok = lambda x: x > 10000  # padding:0
_is_cls = lambda x: (x < 10000) * (x > 0)

# nl
_is_nl = lambda x: _row_idx(x) == 99
_is_nl_cls = lambda x: _cell_idx(x) == 9899
_is_nl_wo_cls = lambda x: (_row_idx(x) == 99) * (_col_idx(x) <= 97)


# tbl
f_is_meta = lambda x: (_row_idx(x)>=98) | (_col_idx(x)>=98)
_is_meta = lambda x: (_row_idx(x)>=98) | (_col_idx(x)>=98)
_is_meta_cls = lambda x: _is_cls(x) & _is_meta(x)
_is_meta_wo_cls = lambda x:_is_tok(x) * (_is_meta(x)>=98)

_is_title = lambda x: x % 10000 == 9898
_is_title_cls = lambda x: x == 9898
_is_title_wo_cls = lambda x: (x > 10000) and (x % 10000 == 9898)

# _is_header = lambda x: (x % 100 == 98) and (x // 100 % 100 <= 97)
_is_header = lambda x: (_row_idx(x)==98)  * (_col_idx(x)<=97)
_is_header_cls = lambda x: (x < 10000) * _is_header(x)
_is_header_wo_cls = lambda x: (x > 10000) * _is_header(x)

_is_row_cls = lambda x: (_is_cls(x)) * (_col_idx(x) == 98)
# cell
_is_cell = lambda x: (_col_idx(x) >= 1) * (_col_idx(x) <= 97) * \
                     (_row_idx(x) >= 1) * (_row_idx(x) <= 97)
_is_cell_tok = lambda x: _is_tok(x) * _is_cell(x)
_is_cell_cls = lambda x: _is_cls(x) * _is_cell(x)


def generate_seg_with_meta_2(args, cols, tbl_id=None,
                           title=None, header=None,
                           noise_num=0, row_wise_fill=True,
                           dgl_backend=False, for_meta=False):
    '''
    :param cols -> List[List[str]]
            ps :
            len(cols)<=98, corresponding to col_id 1..98
            because col_id 99 is left to denote [CLS]
    :param row_wise_fill -> Bool
    :return: tokens -> List[int]; seg -> List[int]
    '''
    # TAB CLS
    has_high_level_cls = True
    if args.has_high_level_cls is False or for_meta:
        has_high_level_cls = False
    if has_high_level_cls:
        tokens = [TAB_CLS_ID]
        seg = [9898]
    else:
        tokens = []
        seg = []
    if title not in [None, '']:
        temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(title))
        if len(temp) == 0:
            pass
        else:
            idx_c = 98; idx_r = 98
            tokens.extend(temp)
            for idx_tok in range(1, len(temp)+1):
                seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)
    if not row_wise_fill:
        for idx_c in range(1, len(cols) + 1):
            if has_high_level_cls:
                idx_r = 98  # fake
                seg.append(100*idx_c+idx_r)
                tokens.append(COL_CLS_ID)
            for idx_r in range(1, len(cols[0]) + 1):
                dataframe = cols[idx_c - 1][idx_r - 1]
                temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))
                if len(temp) == 0:
                    continue
                else:
                    if has_high_level_cls:
                        temp = [CELL_CLS_ID] + temp
                    tokens.extend(temp)
                    for idx_tok in range(0, len(temp)):
                        seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)
    elif row_wise_fill:
        dataframe_max_len = 200
        if not dgl_backend:
            dataframe_max_len = args.seq_len // len(cols)

        if header == [] or header is None:
            if has_high_level_cls:
                for idx_c in range(1, len(cols) + 1):
                    idx_r = 98 # fake
                    seg.append(100*idx_c+idx_r)
                    tokens.append(COL_CLS_ID)
        else:
            for idx_c, col_name in enumerate(header, 1):
                idx_r = 98
                dataframe = col_name
                temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))[
                       :dataframe_max_len - 2]
                if has_high_level_cls:
                    temp = [CLS_ID] + temp
                tokens.extend(temp)
                for idx_tok in range(0, len(temp)):
                    seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)

        for idx_r in range(1, len(cols[0])+1):
            if has_high_level_cls:
                idx_c = 98
                tokens.extend([ROW_CLS_ID])
                seg.extend([idx_c*100 +idx_r])
            for idx_c in range(1, len(cols) + 1):
                dataframe = str(cols[idx_c-1][idx_r-1])
                temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))[:dataframe_max_len-2]
                if len(temp) == 0:
                    continue
                else:
                    if has_high_level_cls:
                        temp = [CELL_CLS_ID] + temp
                    tokens.extend(temp)
                    for idx_tok in range(0, len(temp)):
                        seg.append(idx_tok*10000 + idx_c*100 +idx_r)

    if not dgl_backend:
        tokens = tokens[:args.seq_len]
        seg = seg[:args.seq_len]
        while len(tokens) < args.seq_len:
            tokens.append(PAD_ID)
            seg.append(0)

    if tbl_id is not None:
        assert tbl_id < 100
        seg = list(map(lambda x:100*x+tbl_id, seg))

    for _ in range(noise_num): # two noise
        _i = random.randint(0, len(tokens)-1)
        tokens[_i] = MASK_ID

    return tokens, seg





def generate_seg_with_meta(args, cols, tbl_id=None,
                           title=None, header=None, nl=None,
                           noise_num=0, row_wise_fill=True,
                           # dgl_backend=False
                           ):
    '''
    :param cols -> List[List[str]]
            ps :
            len(cols)<=98, corresponding to col_id 1..98
            because col_id 99 is left to denote [CLS]
    :param row_wise_fill -> Bool
    :return: tokens -> List[int]; seg -> List[int]
    '''
    # TAB CLS
    # has_high_level_cls = True
    # if args.has_high_level_cls is False or for_meta:
    #     has_high_level_cls = False

    # MAGIC_ROW_ID = 98
    # MAGIC_COL_ID = 98

    tokens = []
    seg = []

    if args.high_level_clses == ['ALL']:
        args.high_level_clses = ['TAB', 'COL', 'ROW', 'CELL']
    if args.noise_num:
        noise_num = args.noise_num
    if args.row_wise_fill:
        row_wise_fill = args.row_wise_fill

    # nl
    # import ipdb; ipdb
    if nl not in [None, '', []]:
        idx_r = 99
        if 'NL' in args.high_level_clses:
            tokens.append(NL_CLS_ID)
            seg.append(98*100+idx_r)
        for idx_c in range(1, len(nl) + 1):
            lemma = nl[idx_c - 1]
            lemma_toks = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(lemma))
            tokens.extend(lemma_toks)
            for idx_tok in range(1, len(lemma_toks)+1):
                seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)

    # tab-cls
    if not (cols in (None, [], [[]]) and title in (None, '') and header in ([], None)):
        if 'TAB' in args.high_level_clses:
            tokens.append(TAB_CLS_ID)
            seg.append(101)  # 9898
        # else:
        #     tokens = []
        #     seg = []
    # tab-title
    if title not in [None, '']:
        temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(title))
        if len(temp) == 0:
            pass
        else:
            idx_c = 98; idx_r = 98
            tokens.extend(temp)
            for idx_tok in range(1, len(temp)+1):
                seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)

    # content with header
    if header not in (None, []) or cols not in ('', None, [], [[]], ['']):
        if header not in (None, []) and cols in ('', None, [], [[]], ['']):
            cols = [[]] * len(header)
        if not row_wise_fill: # col_wise
            if 'ROW' in args.high_level_clses:
                for idx_r in range(1, len(cols[0])+1):
                    idx_c = 98
                    tokens.extend([ROW_CLS_ID])
                    seg.extend([idx_c*100 +idx_r])
            for idx_c in range(1, len(cols) + 1):
                # col_cls
                if 'COL' in args.high_level_clses:
                    idx_r = 98  # fake
                    seg.append(100*idx_c+idx_r)
                    tokens.append(COL_CLS_ID)
                # header
                if header not in [[], None]:
                    dataframe = header[idx_c - 1]
                    temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))
                    if len(temp) == 0:
                        continue
                    else:
                        tokens.extend(temp)
                        for idx_tok in range(1, len(temp)+1):
                            seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)
                # cell
                for idx_r in range(1, len(cols[0]) + 1):
                    dataframe = str(cols[idx_c - 1][idx_r - 1])
                    temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))
                    if len(temp) == 0:
                        continue
                    else:
                        if 'CELL' in args.high_level_clses:
                            seg.append(100 * idx_c + idx_r)
                            tokens.append(CELL_CLS_ID)
                        tokens.extend(temp)
                        for idx_tok in range(1, len(temp)+1):
                            seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)
        elif row_wise_fill:
            dataframe_max_len = 200
            if not args.dgl_backend:
                dataframe_max_len = args.seq_len // len(cols)

            # header
            if header == [] or header is None:
                if 'COL' in args.high_level_clses:
                    for idx_c in range(1, len(cols) + 1):
                        idx_r = 98 # fake
                        seg.append(100*idx_c+idx_r)
                        tokens.append(COL_CLS_ID)
            else:
                for idx_c, col_name in enumerate(header, 1):
                    idx_r = 98
                    dataframe = col_name
                    temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))[
                           :dataframe_max_len - 2]
                    if 'COL' in args.high_level_clses:
                        tokens.extend([COL_CLS_ID])
                        seg.extend([idx_c * 100 + idx_r])
                    tokens.extend(temp)
                    for idx_tok in range(1, len(temp)+1):
                        seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)

            # rows
            for idx_r in range(1, len(cols[0])+1):
                # row
                if 'ROW' in args.high_level_clses:
                    # ROW_CLS
                    idx_c = 98
                    tokens.extend([ROW_CLS_ID])
                    seg.extend([idx_c*100 +idx_r])
                for idx_c in range(1, len(cols) + 1):
                    dataframe = str(cols[idx_c-1][idx_r-1])
                    temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))[:dataframe_max_len-2]
                    if len(temp) == 0:
                        continue
                    else:
                        # cell
                        if 'CELL' in args.high_level_clses:
                            tokens.extend([CELL_CLS_ID])
                            seg.extend([idx_c * 100 + idx_r])
                        tokens.extend(temp)
                        for idx_tok in range(1, len(temp)+1):
                            seg.append(idx_tok*10000 + idx_c*100 +idx_r)
    if not args.dgl_backend:
        tokens = tokens[:args.seq_len]
        seg = seg[:args.seq_len]
        while len(tokens) < args.seq_len:
            tokens.append(PAD_ID)
            seg.append(0)

    if tbl_id is not None:
        assert tbl_id < 100
        seg = list(map(lambda x:100*x+tbl_id, seg))

    for _ in range(noise_num): # two noise
        _i = random.randint(0, len(tokens)-1)
        tokens[_i] = MASK_ID

    return tokens, seg




def generate_mask_TODO(seg, mask_mode='cross-wise'):
    '''
    1) In parallel with what I had written in my paper.
    2) Drop using cross-wise AND additional mask

    Difference:
    1) tab-cls 可以看到 col-cls 和 row-cls （但是不能看到 cell-cls）
    2) cell-cls 可以看到 cross 位置的 token
    3) 可选同行或同列可见
    '''

    if seg.dim() == 1:
        seg = seg.unsqueeze(0)

    bz, seq_len = seg.shape
    seg = seg.view(bz, seq_len, 1)
    seg_2 = seg.view(bz, 1, seq_len)

    # import ipdb; ipdb.set_trace()
    # 1. cross part; tok and cell-cls
    if mask_mode=='cross-wise':
        wise_see = (_row_idx(seg) == _row_idx(seg_2)) + (_col_idx(seg) == _col_idx(seg_2))
    if mask_mode=='row-wise':
        wise_see = (_row_idx(seg) == _row_idx(seg_2))
    if mask_mode=='col-wose':
        wise_see = (_col_idx(seg) == _col_idx(seg_2))
    wise_see = wise_see*(~ _is_meta_cls(seg))*(~ _is_meta_cls(seg_2))
    additional_mask = ~ (_is_cell_cls(seg_2) * (_cell_idx(seg) != _cell_idx(seg_2)))
    wise_see = wise_see * additional_mask

    #
    # 2. hier part
    # 2.1. down_up:
    tab_see_1 = (_is_title_cls(seg) * _is_tok(seg_2))
    col_row_see_1 = (_is_col_cls(seg) + _is_row_cls(seg)) * \
                     _is_tok(seg_2) * \
                     ( (_row_idx(seg) == _row_idx(seg_2)) + (_col_idx(seg) == _col_idx(seg_2))
    )


    # 2.2. up_down and parallel
    col_row_see_2 = (_is_col_cls(seg) + _is_row_cls(seg)) * \
                     _is_meta(seg_2) * \
                     ( (_row_idx(seg) == _row_idx(seg_2)) + (_col_idx(seg) == _col_idx(seg_2))
    )
    cell_see = _is_cell_cls(seg) * (_is_row_cls(seg) + _is_header_cls(seg_2))\
               *((_row_idx(seg) == _row_idx(seg_2)) + (_col_idx(seg) == _col_idx(seg_2)))

    m = (wise_see + tab_see) * additional_mask
    return m.float().unsqueeze(1)




def generate_mask_2(seg, mask_mode='cross-wise'):
    '''
    1) In parallel with what I had written in my paper.
    2) Drop using cross-wise AND additional mask

    Difference:
    1) tab-cls 可以看到 col-cls 和 row-cls （但是不能看到 cell-cls）
    2) cell-cls 可以看到 cross 位置的 token
    3) 可选同行或同列可见
    '''

    if seg.dim() == 1:
        seg = seg.unsqueeze(0)

    bz, seq_len = seg.shape
    seg = seg.view(bz, seq_len, 1)
    seg_2 = seg.view(bz, 1, seq_len)

    # import ipdb; ipdb.set_trace()
    # 1. cross part
    if mask_mode=='cross-wise':
        wise_see = (_row_idx(seg) == _row_idx(seg_2)) + (_col_idx(seg) == _col_idx(seg_2))

    if mask_mode=='row-wise':
        wise_see = (_row_idx(seg) == _row_idx(seg_2))
    if mask_mode=='col-wose':
        wise_see = (_col_idx(seg) == _col_idx(seg_2))

    # 2. hier part
    # 2.1. down_up:
    tab_see = (_is_title_cls(seg) * _is_tok(seg_2))

    # 2.2. 阻断 cell-cls到处被看到
    additional_mask = ~ (_is_cell_cls(seg_2) * (_cell_idx(seg) != _cell_idx(seg_2)))

    m = (wise_see + tab_see) * additional_mask
    return m.float().unsqueeze(1)


def generate_mask(seg, mask_mode='cross-wise'):
    '''
    1) In parallel with what I had written in my paper.
    2) Drop using cross-wise AND additional mask

    Difference:
    1) tab-cls 可以看到 col-cls 和 row-cls （但是不能看到 cell-cls）
    2) cell-cls 可以看到 cross 位置的 token
    3) 可选同行或同列可见
    '''

    if seg.dim() == 1:
        seg = seg.unsqueeze(0)

    bz, seq_len = seg.shape
    seg = seg.view(bz, seq_len, 1)
    seg_2 = seg.view(bz, 1, seq_len)

    # if mask_mode == 'cross-wise':
    wise_see = (_row_idx(seg) == _row_idx(seg_2)) + (_col_idx(seg) == _col_idx(seg_2))
    tab_see = (_is_title_cls(seg) * _is_tok(seg_2))
    '''
        包含了十字上下文、col/row-cls向下包含关系、层级之间的链接
        需要去除：
            0. 向上看到太多（只有token到列而已啦）
            1. 向下看到的太多 cls
            2. token 向上看到的太多 cls: 同行、同列其它cell-cls
            3.1. 水平位置看到的 cls (cell_cls与其它cell_cls, row_cls与row_cls)
            3.2. cell_cls与其它cell_cls(row/col-cls保持平行之间的可见) 
            
            消除 1 2 3.1
            (1) 消除cell_cls被同cell_idx之外元素的可见 
                2被消除 3.2被消除 1剩下tab看col 
            (2)  消除 tab、col、row看见col、row
                tab_col_row 不能看见 col_row
            
            消除 1 2 3.2
            (1) 消除cell_cls被同cell_idx之外元素的可见 
                2被消除 3.2被消除 1剩下tab看col 
            (2)  消除 tab看见col、row
    '''
    # import ipdb; ipdb.set_trace()
    additional_mask = ~ (_is_cell_cls(seg_2) * (_cell_idx(seg) != _cell_idx(seg_2)))
    additional_mask_2 = ~ (_is_meta_cls(seg) * (_is_row_cls(seg_2) | _is_header_cls(seg_2)))
    additional_mask_2 = additional_mask_2 + torch.eye(*additional_mask_2[0].shape, device=additional_mask.device).bool()

    m = (wise_see + tab_see) * additional_mask * additional_mask_2
    return m.float().unsqueeze(1)


