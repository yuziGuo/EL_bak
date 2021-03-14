import torch
from col_spec_yh.constants import *
from col_spec_yh.store_utils import decode_sqlite_spider_tbls_in_rows_with_metadata
from demos.utils import get_args_minimal as get_args
import random

_tbl_idx = lambda x: x % 100

_row_idx = lambda x: x % 100
_col_idx = lambda x: x % 10000 // 100
_cell_idx = lambda x: x % 10000

_is_tok = lambda x: x > 10000  # padding:0
_is_cls = lambda x: x < 10000 and x > 0

# nl
_is_nl = lambda x: _row_idx(x) == 99
_is_nl_cls = lambda x: _cell_idx(x) == 9899
_is_nl_wo_cls = lambda x: (_row_idx(x) == 99) * (_col_idx(x) <= 97)


# tbl
f_is_meta = lambda x: _row_idx(x)>=98
_is_meta = lambda x: _row_idx(x)>=98
_is_meta_wo_cls = lambda x:_is_tok(x) * _row_idx(x)>=98

_is_title = lambda x: x % 10000 == 9898
_is_title_cls = lambda x: x == 9898
_is_title_wo_cls = lambda x: (x > 10000) and (x % 10000 == 9898)

# _is_header = lambda x: (x % 100 == 98) and (x // 100 % 100 <= 97)
_is_header = lambda x: (_row_idx(x)==98)  * (_col_idx(x)<=97)
_is_header_cls = lambda x: (x < 10000) * _is_header(x)
_is_header_wo_cls = lambda x: (x > 10000) * _is_header(x)

_is_row_cls = lambda x: (_is_cls(x)) * (_col_idx(x) == 98)
# cell
_is_cell = lambda x: _col_idx(x) >= 1 * _col_idx(x) <= 97 * \
                     _row_idx(x) >= 1 * _row_idx(x) <= 97
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
                           dgl_backend=False):
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

    # nl
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
            seg.append(9898)
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
        if not row_wise_fill:
            for idx_r in range(1, len(cols[0])+1):
                if 'ROW' in args.high_level_clses:
                    idx_c = 98
                    tokens.extend([ROW_CLS_ID])
                    seg.extend([idx_c*100 +idx_r])
            for idx_c in range(1, len(cols) + 1):
                if 'COL' in args.high_level_clses:
                    idx_r = 98  # fake
                    seg.append(100*idx_c+idx_r)
                    tokens.append(COL_CLS_ID)
                if header not in [[], None]:
                    dataframe = header[idx_c - 1]
                    temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))
                    if len(temp) == 0:
                        continue
                    else:
                        tokens.extend(temp)
                        for idx_tok in range(0, len(temp)):
                            seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)

                for idx_r in range(1, len(cols[0]) + 1):
                    dataframe = cols[idx_c - 1][idx_r - 1]
                    temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))
                    if len(temp) == 0:
                        continue
                    else:
                        if 'CELL' in args.high_level_clses:
                            temp = [CELL_CLS_ID] + temp
                        tokens.extend(temp)
                        for idx_tok in range(0, len(temp)):
                            seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)
        elif row_wise_fill:
            dataframe_max_len = 200
            if not dgl_backend:
                dataframe_max_len = args.seq_len // len(cols)

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
                        temp = [COL_CLS_ID] + temp
                    tokens.extend(temp)
                    for idx_tok in range(0, len(temp)):
                        seg.append(idx_tok * 10000 + idx_c * 100 + idx_r)

            for idx_r in range(1, len(cols[0])+1):
                if 'ROW' in args.high_level_clses:
                    idx_c = 98
                    tokens.extend([ROW_CLS_ID])
                    seg.extend([idx_c*100 +idx_r])
                for idx_c in range(1, len(cols) + 1):
                    dataframe = str(cols[idx_c-1][idx_r-1])
                    temp = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(dataframe))[:dataframe_max_len-2]
                    if len(temp) == 0:
                        continue
                    else:
                        if 'CELL' in args.high_level_clses:
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







