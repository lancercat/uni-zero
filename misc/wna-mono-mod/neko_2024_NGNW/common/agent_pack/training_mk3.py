from osocrNG.configs.typical_agent_setups.data_ferrier import get_osocr_data_ferrier_mk2
from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from neko_2024_NGNW.common.namescope import mod_names,agent_var_names

class neko_training_common_mk3:
    MN=mod_names;
    VAN=agent_var_names;
    def get_data_ferrier(this, incoming_q):
        return get_osocr_data_ferrier_mk2(incoming_q, {
            RN.IMAGE: this.VAN.RAW_IMG_NAME,
            RN.LABEL: this.VAN.RAW_LABEL_NAME,
            RN.UID: this.VAN.RAW_IMG_TAG,
            RN.SIZE: this.VAN.RAW_SIZE_NAME,
            RN.ANCHOR: this.VAN.RAW_ANCHOR_NAME,
        })