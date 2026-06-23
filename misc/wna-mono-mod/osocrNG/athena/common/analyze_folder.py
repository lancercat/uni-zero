
import glob
import os

from osocrNG.athena.common.quickptg1 import prepare_pt,prepare_pt_ng


def bootstrap_folder(root,dst,lang,pfix="*.jpg"):
    ptfile,_=prepare_pt(os.path.join(root,lang));
    sfolder=os.path.join(root,lang);
    dfolder=os.path.join(dst,os.path.basename(lang),"results");
    os.makedirs(dfolder,exist_ok=True);
    files = glob.glob(os.path.join(sfolder, pfix));
    return files,ptfile,sfolder,dfolder;


def bootstrap_folder_NG(root,dst,lang,pfix="*.jpg"):
    ptfile,v2ptfolder=prepare_pt_ng(os.path.join(root,lang));
    sfolder=os.path.join(root,lang);
    dfolder=os.path.join(dst,os.path.basename(lang),"results");
    os.makedirs(dfolder,exist_ok=True);
    files = glob.glob(os.path.join(sfolder, pfix));
    return files,v2ptfolder,sfolder,dfolder;
