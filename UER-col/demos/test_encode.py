import torch
from demos.samples.sample_mini_tables import table_a, table_b
from utils import get_args, load_or_initialize_parameters
from col_spec_yh.model import TabEncoder
from col_spec_yh.encode_utils import generate_seg
from col_spec_yh.encode_utils import generate_mask

args = get_args()

# model
ta_encoder = TabEncoder(args)
load_or_initialize_parameters(args, ta_encoder)

# data
tokens_0, seg_0 = generate_seg(args, table_a, row_wise_fill=True)
tokens_1, seg_1 = generate_seg(args, table_a, row_wise_fill=True)
src = torch.LongTensor([tokens_0, tokens_1])
seg = torch.LongTensor([seg_0, seg_1])
# mask = generate_mask_crosswise(seg)

_ = ta_encoder(src, seg)
import ipdb; ipdb.set_trace()

