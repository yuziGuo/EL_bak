from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask_crosswise
from col_spec_yh.model import TabEncoder

from demos.samples.sample_mini_tables import table_a, table_b
from demos.utils import get_args
from demos.utils import load_or_initialize_parameters

import torch
from col_spec_yh.store_utils import decode_and_verify_spider_file

args = get_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
tab_file = 'demos/samples/sample_file_type0-1.tb'
tab_cols_list = decode_and_verify_spider_file(tab_file)
seg_list = []
src_list = []
for tab_col in tab_cols_list:
    tokens, seg = generate_seg(args, tab_col, row_wise_fill=True)
    seg_list.append(seg)
    src_list.append(tokens)
seg_batch = torch.LongTensor(seg_list)
src_batch = torch.LongTensor(src_list)
# mask_batch = generate_mask_crosswise(seg_batch)
# mask.shape: torch.Size([10, 1, 64, 64])

# model
args.pooling = 'avg-cell-seg'
ta_encoder = TabEncoder(args)
load_or_initialize_parameters(args, ta_encoder)
ta_encoder = ta_encoder.to(args.device)

# encode
with torch.no_grad():
    src_batch = src_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    _ = ta_encoder.encode(src_batch, seg_batch, option='first-column')  # [bz, emb_size]
    # _ = ta_encoder.encode(src_batch, seg_batch, option='columns')  # [bz, col_num, emb_size]
    import ipdb; ipdb.set_trace()


# _ = ta_encoder.encode(src, seg, option='table', pooling='avg-cell-seg')  # [bz, emb_size]
