

import os.path

from osocrNG.data_utils.common_data_presets.dspathsNG import get_mltkr_path,get_mltjphv_path,get_mltjp_path

from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder


class moostr_ostr_nomask:
    @classmethod
    def arm_osocr_test_jpn_full(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjp_path];
        NMS = ["JPN"];
        metadict_ = {
            "JPN-GZSL":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmlt"),
                 "case_sensitive": False,
                 "has_unk": False,
                 },
            "JPN-OSR":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmltch_osr"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPN-GOSR":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmltch_nohirakata"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPN-OSTR":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmltch_kanji"),
                 "case_sensitive": False,
                 "has_unk": True,
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_im_label_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];
        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict


    @classmethod
    def arm_osocr_test_jpnhv_gosr(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GOSR":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmlthv_nohirakata"),
                 "case_sensitive": False,
                 "has_unk": True,
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_im_label_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict


    @classmethod
    def arm_osocr_test_jpnhv_full(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss we need the model to pretend not seeing these characters
                 },
            "JPNHV-OSR":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmlthv_osr"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPNHV-GOSR":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmlthv_nohirakata"),
                 "case_sensitive": False,
                 "has_unk": True,
                 },
            "JPNHV-OSTR":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmlthv_kanji"),
                 "case_sensitive": False,
                 "has_unk": True,
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_im_label_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict


    @classmethod
    def arm_osocr_test_jpnhv_gzsl(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltjphv_path];
        NMS = ["JPNHV"];
        metadict_ = {
            "JPNHV-GZSL":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabjpmlthv"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss we need the model to pretend not seeing these characters
                 }
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_im_label_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict


    @classmethod
    def arm_osocr_test_kr_full(cls, meta_dict, data_dict, test_dict, dataroot, v2h=-9):
        FNS = [get_mltkr_path];
        NMS = ["KR"];
        metadict_ = {
            "KR-GZSL":
                {"meta_path": os.path.join(dataroot, "dicts_v3", "dabkrmlt"),
                 "case_sensitive": False,
                 "has_unk": False
                 # There are some datasets just ignores unknown chracters, e.g "999-123456" will be annotated as "999123456".
                 # For these test dss (mainly IIT5k ) we need the model to pretend not seeing these characters as close-set conterparts do.
                 },
        };
        for m in metadict_:
            assert (m not in meta_dict);
            meta_dict[m] = metadict_[m];
        data_dict_ = {}
        for d, n in zip(FNS, NMS):
            data_dict_[n] = neko_im_label_lmdb_holder({"root": d(dataroot), "vert_to_hori": v2h})
            data_dict[n] = data_dict_[n];

        for d in data_dict_:
            for m in metadict_:
                test_dict[d + "-" + m] = {
                    "data": d,
                    "meta": m,
                }
        return meta_dict, data_dict, test_dict
