import os;
def get_nips14(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'NIPS2014');
def get_cvpr16(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'CVPR2016');

def get_nips14sub(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'NIPS2014sub');
def get_cvpr16sub(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'CVPR2016sub');
def get_iiit5k(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IIIT5k_3000');
def get_cute(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'CUTE80');
def get_IC03_867(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IC03_867');
def get_IC13_1015(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"IC13_1015")
def get_IC15_2077(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IC15_2077');
def get_IC15_1811(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'IC15_1811');
def get_SVT(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'SVT');
def get_SVTP(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'SVTP');


def get_lsvt_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'lsvtdb');
def get_ctw(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'ctw_fslchr');
def get_art_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrdb');


def get_ctw2k(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"ctw2kseen")

def get_ctw2kus(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"ctw2kunseen_500")

def get_artK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrdbseen');
def get_mlt_chlatK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltchlatdbseen');
def get_lsvtK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'lsvtdbseen');
def get_ctwK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'ctw_seen');
def get_rctwK_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'rctwdb_seen');


def get_mlt_chlat_path(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltchlatdb');
def get_mlt_chlatval(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltchlatdbval');
def get_artvalseen(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrvaldbseen');
def get_artval(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrvaldb');
def get_8mtr(root="/home/lasercat/ssddata/"):
    ds=[
        os.path.join(root,'8mtr_1'),
        os.path.join(root,'8mtr_2'),
        os.path.join(root,'8mtr_3'),
    ]
    return ds;
def get_qhbcsvtr(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'qhbcsvtr');


def get_docuevalseen(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'docstchsevalseen');
def get_mlt_artvalunseen(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'arttrvaldbunseen');
def get_mlt_jpval(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'mltvaljpdb');
def get_ulsvta(root="/home/lasercat/ssddata/"):
    return os.path.join(root,'lsvtdbunseenA');

def get_fudanvi_scene_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"scene","scene_train");
def get_fudanvi_scene_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"scene","scene_val");
def get_fudanvi_scene_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"scene","scene_test");

def get_fudanvi_web_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"web","web_train");
def get_fudanvi_web_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"web","web_val");
def get_fudanvi_web_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"web","web_test");

def get_fudanvi_document_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"document","document_train");
def get_fudanvi_document_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"document","document_val");
def get_fudanvi_document_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"document","document_test");

def get_fudanvi_hwdb_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"handwriting_hwdb_train");
def get_fudanvi_hwdb_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"handwriting_hwdb_val");
def get_fudanvi_hwdb_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root, 'fudanvi',"handwriting_ic13_test");



def get_mj_tr_abi(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"MJ-abi","MJ_train");
def get_mj_te_abi(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"MJ-abi","MJ_test");
def get_mj_val_abi(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"MJ-abi","MJ_valid");
def get_st_abi(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"ST-abi");

def get_istr_bengali_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","bengali_test");
def get_istr_bengali_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","bengali_val");
def get_istr_gujarati_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","gujarati_train");
def get_istr_hindi_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","hindi_test");
def get_istr_hindi_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","hindi_val");
def get_istr_kannada_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","kannada_train");
def get_istr_malayalam_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","malayalam_test");
def get_istr_malayalam_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","malayalam_val");
def get_istr_marathi_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","marathi_train");
def get_istr_punjabi_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","punjabi_test");
def get_istr_punjabi_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","punjabi_val");
def get_istr_tamil_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","tamil_train");
def get_istr_telugu_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","telugu_test");
def get_istr_telugu_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","telugu_val");
def get_istr_bengali_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","bengali_train");
def get_istr_gujarati_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","gujarati_test");
def get_istr_gujarati_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","gujarati_val");
def get_istr_hindi_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","hindi_train");
def get_istr_kannada_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","kannada_test");
def get_istr_kannada_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","kannada_val");
def get_istr_malayalam_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","malayalam_train");
def get_istr_marathi_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","marathi_test");
def get_istr_marathi_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","marathi_val");
def get_istr_punjabi_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","punjabi_train");
def get_istr_tamil_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","tamil_test");
def get_istr_tamil_val(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","tamil_val");
def get_istr_telugu_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"istr","telugu_train");
def get_hhd_train(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"hhd","train_raw");
def get_hhd_test(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"hhd","test_rand");
def get_hhd_test_1800ad(root="/home/lasercat/ssddata/"):
    return os.path.join(root,"hhd","test_18th");

INDIC_STR12=["Assamese","Bengali","Gujarati","Hindi","Kannada","Malayalam","Marathi","Meitei_Manipuri","Odia","Punjabi","Tamil","Telugu","Urdu"];
def get_indicstr12_synth(lang,split,root="/home/lasercat/ssddata"):
    return os.path.join(root, "indicstr12synth", lang+"_"+split);
