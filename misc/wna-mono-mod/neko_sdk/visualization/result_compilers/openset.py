from neko_2020nocr.dan.utils import neko_os_Attention_AR_counter,neko_oswr_Attention_AR_counter

from neko_sdk.ocr_modules.charset.jpn_filters import get_jpn_filters_compact
from neko_sdk.ocr_modules.charset.no_filter import get_no_filters
from neko_sdk.visualization.result_compilers.compiler import compile_alot


def compile_jp(root,modnames,workingdir,devs,branch,methods,codenames,threshs,case_sensitive=False,dwgood=1280,dwbad=640):
    filters=get_jpn_filters_compact();
    no_filters=get_no_filters();
    dss=["JAP_lang"]
    ds_codenames=["JP"];
    protocols=["GZSL","OSR","GOSR","OSTR"];
    counters=[neko_os_Attention_AR_counter,
              neko_oswr_Attention_AR_counter,
              neko_oswr_Attention_AR_counter,
              neko_oswr_Attention_AR_counter];
    filterss=[
        filters,
        no_filters,
        no_filters,
        no_filters
    ]
    rdic, cmds    =compile_alot(root,dss,ds_codenames,modnames,workingdir,devs,branch,methods,codenames,threshs,protocols,counters,filterss,case_sensitive,dwgood,dwbad);
    return rdic,cmds;
if __name__ == '__main__':

    ROOT="/run/media/lasercat/ssddata/project_290_dump/"
    BRANCH="fullXL"

    THRESH = [10, 8, 5, 3, 0];


    METHODS=["openXL_ssr_r3g3_mpfS_DT48_D10k_3sp_pm_d05"];
    DEVS = ["MEOWS-ZeroDimension"]

    MODNAME = ["base_chs_prototyper"];
    CODENAME = ["fullXL"];
    WDIR="/run/media/lasercat/ssddata/project_290_dump/G1"
    compile_jp(ROOT,MODNAME,WDIR,DEVS,BRANCH,METHODS,CODENAME,THRESH,dwgood=720,dwbad=360);
    compile_alot(ROOT,MODNAME,)