# A1=AA,no-glp, decoupled anchor system
# for base model, it uses v5-2 anchor setup, which has less anchor overlaps (faster training, that is).

from neko_2024_NGNW.nets_v6.base.modfactory import neko_wna_v6_base_no_pad as modf
from neko_2024_NGNW.nets_v6.no_delay.agtfactory import pgroute_only_agent_factory_v6_no_delay_no_bw_ohem01E as agtf


from neko_2024_NGNW.nets_v6.anchors import get_wna_v6_32_anchor_1h1v1r_nedmix as acfg
from neko_2024_NGNW.nets_v6.anchors import get_wna_v6_dcfg_1h1v1r_2_05_smol as trdcfg
from osocrNG.data_utils.common_data_presets_mk3.moostr_nonmask import moostr_mk3_data_factoryAAF as datf
