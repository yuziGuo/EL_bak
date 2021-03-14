from utils import get_args, load_or_initialize_parameters
from uer.models.tab_encoder import TabEncoder

args = get_args()
ta_encoder = TabEncoder(args)
load_or_initialize_parameters(args, ta_encoder)
# import ipdb; ipdb.set_trace()
