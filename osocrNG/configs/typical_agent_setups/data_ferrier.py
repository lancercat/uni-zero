
from osocrNG.modular_agents_ocrNG.ocr_data_agents.neko_osocr_data_ferrier import neko_ocr_data_ferrier,neko_ocr_data_ferrier_mk2
def get_osocr_data_ferrier(queue_name,image_name,beacon_name,size_name,bmask_name,label_name):
    return {
        "agent":neko_ocr_data_ferrier,
        "params":{
            "iocvt_dict": {
                "queue_name":queue_name,
                "image_name":image_name,
                "beacon_name":beacon_name,
                "bmask_name":bmask_name,
                "label_name":label_name,
                "size_name":size_name,
            },
            "modcvt_dict": {
            }
        }
    }
def get_osocr_data_ferrier_mk2(queue_name,iomap):
    return {
        "agent":neko_ocr_data_ferrier_mk2,
        "params":{
            "iomap":iomap,
            "iocvt_dict": {
                "queue_name":queue_name,
            },
            "modcvt_dict": {
            }
        }
    }
def get_osocr_data_rotator(queue_name,image_name,beacon_name,size_name,bmask_name,label_name):
    return {
        "agent":neko_ocr_data_ferrier,
        "params":{
            "iocvt_dict": {
                "queue_name":queue_name,
                "image_name":image_name,
                "beacon_name":beacon_name,
                "bmask_name":bmask_name,
                "label_name":label_name,
                "size_name":size_name,
            },
            "modcvt_dict": {
            }
        }
    }
