from neko_2022_soai_zero.visualization.result_compilers.compiler import compile_alot

from neko_2020nocr.dan.utils import neko_os_Attention_AR_counter
from neko_sdk.ocr_modules.charset.no_filter import get_no_filters


def compile_mjst(root,modnames,workingdir,dev,branch,methods,codenames,threshs,case_sensitive=False,dwgood=1280,dwbad=640):
    no_filters=get_no_filters();
    dss=["IIIT5k","CUTE","SVT","IC03","IC13"]
    ds_codenames=["IIIT","CUTE","SVT","ICA","ICB"];
    protocols=["closeset_benchmarks"];
    counters=[neko_os_Attention_AR_counter];
    filterss=[
        no_filters,
    ]
    rdic, cmds =compile_alot(root,dss,ds_codenames,modnames,workingdir,dev,branch,methods,codenames,threshs,protocols,counters,filterss,case_sensitive,dwgood,dwbad);
    return rdic,cmds;
if __name__ == '__main__':

    ROOT="/run/media/lasercat/ssddata/all_283/"
    DEV=["MEOWS-Gamarket2","MEOWS-MegaDimension"]
    BRANCH="OR"
    METHODS=["close_basemodel_prec_shufIN3_g3kDPs_cycanof_aa_p2","close_basemodelXL512_prec_shufIN3_g3kDPs_cycanof_aa_p3"];
    THRESH = [10, 8, 5, 3, 0];
    MODNAME = [ "shuf3_mjst_prototyper","shuf3_mjst_prototyper"];
    CODENAME = ["full","fullXL"];
    WDIR="/run/media/lasercat/ssddata/283_paper/G3"
    compile_mjst(ROOT,MODNAME,WDIR,DEV,BRANCH,METHODS,CODENAME,THRESH)