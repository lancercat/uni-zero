from neko_2020nocr.dan.utils import neko_os_Attention_AR_counter

from neko_sdk.ocr_modules.charset.jpn_filters import get_jpn_filters_compact
from neko_sdk.visualization.result_compilers.compiler import compile_alot


def compile_gzsl(root,modnames,workingdir,dev,branch,methods,codenames,threshs,case_sensitive=False,dwgood=1280,dwbad=640):
    dss=["JAP_lang"]
    ds_codenames=["JP"];
    protocols=["GZSL"];
    counters=[neko_os_Attention_AR_counter];
    filterss=[get_jpn_filters_compact()
    ]
    rdic, cmds =compile_alot(root,dss,ds_codenames,modnames,workingdir,dev,branch,methods,codenames,threshs,protocols,counters,filterss,case_sensitive,dwgood,dwbad);
    return rdic,cmds;
if __name__ == '__main__':
    BRANCH = "OR"

    ROOT="/run/media/lasercat/ssddata/all_283/"
    DEV=["MEOWS-ZeroDimension"]
    METHODS=["open_basemodelXL512_prec_shufIN3_g3kDPs_cycanof_aa"];
    MODNAME = [ "shuf3_chs_prototyper"];
    CODENAME = ["fullXL"];


    ROOT="/run/media/lasercat/ssddata/all_283/"
    DEV=["MEOWS-ZeroDimension"]
    METHODS=["open_basemodelXL512_prec_shufIN3_g3kDPs_cycanof_aa"];
    MODNAME = [ "shuf3_chs_prototyper"];
    CODENAME = ["fullXL"];


    THRESH = [10, 8, 5, 3, 0];
    WDIR="/run/media/lasercat/ssddata/283_paper/G3"
    compile_gzsl(ROOT,MODNAME,WDIR,DEV,BRANCH,METHODS,CODENAME,THRESH)