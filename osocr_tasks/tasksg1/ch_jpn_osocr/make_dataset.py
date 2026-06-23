import os

from neko_sdk.lmdb_wrappers.splitds import shfilter, harvast_cs
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.etc_cset import latin62, korean
from neko_sdk.ocr_modules.charset.symbols import symbol;
from neko_sdk.ocr_modules.lmdbcvt.artcvt import make_art_lmdb
from neko_sdk.ocr_modules.lmdbcvt.ctwcvt import make_ctw
from neko_sdk.ocr_modules.lmdbcvt.lsvtcvt import makelsvt
from neko_sdk.ocr_modules.lmdbcvt.mltlike import make_mlt_valhindi, \
    make_mlt_valkr, make_mlt_train_kr, make_mlt_train_bangla, \
    make_mlt_train_chlat, make_mlt_train_jp
from neko_sdk.ocr_modules.lmdbcvt.mltlike import make_rctw_train  # well thay look alike :-)
from osocr_tasks.tasksg1.ch_jpn_osocr.dict3817SCmetafy import make_chlat_sc_dict
from osocr_tasks.tasksg1.ch_jpn_osocr.dict3817WTmetafy import make_chlat_wt_dict
from osocr_tasks.tasksg1.ch_jpn_osocr.dictchslatkrMCmetafy import make_chlatkr_mc_dict
from osocr_tasks.tasksg1.dscs import get_ds;
from osocr_tasks.tasksg1.dscs import makept

ROOT="/run/media/lasercat/writebuffer/deploy/"

PATHS={
    "artroot":ROOT+"/art",
    "ctwtrgtroot":ROOT+"/ctw/gtar/train.jsonl",
    "ctwtrimroot": ROOT+"/ctw/jpgs",
    "mltroot": ROOT+"/mlt/real",
    "mltsynthchroot":ROOT+"/mlt/Chinese",
    "rctwtrroot": ROOT+"/rctw_train/train",
    "lsvttrjson":ROOT+"/lsvt/train_full_labels.json",
    "lsvttrimgs":ROOT+"/lsvt/imgs/",
    "dictroot": ROOT+"/deployedlmdbs/dicts",
    "desroot":ROOT+"/deployedlmdbs",
    "fntpath": [ROOT + "/NotoSansCJK-Regular.ttc"],
    "bafntpath":[ROOT + "/NotoSansBengali-Regular.ttf"],
}

def make_trch_tejpn_datasets():
    rawart=os.path.join(PATHS["desroot"],"artdb");
    rawrctwtr=os.path.join(PATHS["desroot"], "rctwtrdb");
    rawlsvt=os.path.join(PATHS["desroot"], "lsvtdb");
    rawmltchlat=os.path.join(PATHS["desroot"], "mlttrchlat");
    rawmltkr=os.path.join(PATHS["desroot"], "mlttrkr");
    rawbangla=os.path.join(PATHS["desroot"], "mlttrbengala")
    rawctw=os.path.join(PATHS["desroot"],"ctwdb");

    sart=os.path.join(PATHS["desroot"],"artdb_seen");
    srctwtr=os.path.join(PATHS["desroot"], "rctwtrdb_seen");
    slsvt=os.path.join(PATHS["desroot"], "lsvtdb_seen");
    smltchlat=os.path.join(PATHS["desroot"], "mlttrchlat_seen");
    sctw=os.path.join(PATHS["desroot"],"ctwdb_seen");
    skr=os.path.join(PATHS["desroot"],"mltkrdb_seen");
    sbe=os.path.join(PATHS["desroot"],"mltbedb_seen");
    sar=os.path.join(PATHS["desroot"],"mltardb_seen");

    rawtr=[rawart,rawrctwtr,rawlsvt,rawmltchlat,rawctw];
    fintr=[sart,srctwtr,slsvt,smltchlat,sctw];

    raw_eval = os.path.join(PATHS["desroot"], "mlttrjp")
    heval=os.path.join(PATHS["desroot"], "mlttrjp_hori")
    hkreval=os.path.join(PATHS["desroot"], "mlttrkr_hori")

    raw_evalh = os.path.join(PATHS["desroot"], "mlttrhindi")
    hevalh=os.path.join(PATHS["desroot"], "mlttrhn_hori")


    raw_evalb = os.path.join(PATHS["desroot"], "mlttrbegali")
    hevalb=os.path.join(PATHS["desroot"], "mlttrbe_hori")
    trdspath=os.path.join(PATHS["dictroot"], "dab3791WT.pt")
    trdsscpath=os.path.join(PATHS["dictroot"], "dab3791SC.pt")
    trdskrpath=os.path.join(PATHS["dictroot"], "dabclkMC.pt")
    trdsbapath=os.path.join(PATHS["dictroot"], "dabclbaMC.pt")

    #
    make_art_lmdb(PATHS["artroot"],
                  rawart);
    make_rctw_train(PATHS["rctwtrroot"],
                    rawrctwtr);
    make_ctw(PATHS["ctwtrgtroot"],
             PATHS["ctwtrimroot"],
             rawctw);
    makelsvt(PATHS["lsvttrjson"],
             PATHS["lsvttrimgs"],
             rawlsvt);
    shfilter(rawlsvt,latin62.union(t1_3755),slsvt)
    make_mlt_train_chlat(
        PATHS["mltroot"],
        rawmltchlat
    );

    make_mlt_train_jp(PATHS["mltroot"],
                      raw_eval);
    make_mlt_train_bangla(PATHS["mltroot"],rawbangla);

    make_mlt_train_kr(PATHS["mltroot"],
                   rawmltkr);
    shfilter(rawmltkr, latin62.union(t1_3755).union(korean), hkreval);
    makept(hkreval,
           PATHS["fntpath"],
           os.path.join(PATHS["dictroot"], "dabkrmlt.pt"),
           latin62,
           symbol.union(t1_3755)
           )
    #the mlt annotation is a little bit messy, some korean scripts are mixed in
    jplatchars=list(set(get_ds(raw_eval)).difference(korean.union(symbol)));
    # Like we said, we do not handle vertical scripts(It breaks batching and adds more effort on coding to transpose them. )
    shfilter(raw_eval,jplatchars,heval);
    makept(heval,
           PATHS["fntpath"],
           os.path.join(PATHS["dictroot"], "dabjpmlt.pt"),
           latin62,
           symbol.union(korean)
           )

    os.makedirs(PATHS["dictroot"],exist_ok=True);

    make_chlat_wt_dict(PATHS["fntpath"],
                       trdspath);

    make_chlat_sc_dict(PATHS["fntpath"],
                       trdsscpath);
    make_chlatkr_mc_dict(PATHS["fntpath"],
                       trdskrpath);
    make_mlt_train_bangla(PATHS["mltroot"], rawbangla);
    bcs = harvast_cs(rawbangla)
    # bcs=bcs.difference(symbol).difference(latin62);
    shfilter(rawbangla,bcs,sbe);

    makept(rawbangla,
           PATHS["bafntpath"],
           os.path.join(PATHS["dictroot"], "dabbemlt.pt"),
           latin62,
           symbol.union(korean)
           )
    make_mlt_train_bangla(PATHS["fntpath"],
                       trdsbapath);
    bcs = harvast_cs(rawbangla);
    bcs=bcs.difference(symbol).difference(latin62);
    shfilter(rawbangla,bcs,sbe);
    make_chlatkr_mc_dict(PATHS["fntpath"],trdskrpath,bcs);

    pass;

    # removing all vertical clips in  testing set
    # These are not what we aim to solve in this set of papers.
    # Well they will first settle the horizontal ones and then go back to this.
    # After all I am graduating... Even Hu knows not where the cat is jumping to.

    make_mlt_valkr(PATHS["mltroot"])
    # hfilter(raw_eval,heval);
    # hfilter(raw_evalh,hevalh);
    make_mlt_valhindi(PATHS["mltroot"],
                   raw_evalh);
    hindlatchars=list(set(get_ds(raw_evalh)).difference(korean.union(symbol)));
    # Like we said, we do not handle vertical scripts(It breaks batching and adds more effort on coding to transpose them. )
    shfilter(raw_evalh,hindlatchars,hevalh);
    makept(hevalh,
           ["/home/lasercat/ssddata/fonts-main/Noto-hinted/NotoSansDevanagari-Regular.ttf"],
           os.path.join(PATHS["dictroot"], "dabhnmlt.pt"),
           latin62,
           symbol.union(korean)
           )
    for s, d in zip(rawtr, fintr):
        shfilter(s,latin62.union(t1_3755),d);

    # removing all vertical clips in training set and clips with unseen characters.
    # These are not what aims to solve in this paper.

if __name__ == '__main__':
    make_trch_tejpn_datasets();
