import os
import sys

from neko_sdk.lmdb_wrappers.splitds import shfilter, harvast_cs
from neko_sdk.ocr_modules.charset.chs_cset import t1_3755
from neko_sdk.ocr_modules.charset.etc_cset import latin62, korean
from neko_sdk.ocr_modules.charset.symbols import symbol;
from neko_sdk.ocr_modules.lmdbcvt.artcvt import make_art_lmdb
from neko_sdk.ocr_modules.lmdbcvt.ctwcvt import make_ctw
from neko_sdk.ocr_modules.lmdbcvt.lsvtcvt import makelsvt
from neko_sdk.ocr_modules.lmdbcvt.mltlike import \
    make_mlt_train_Korea, make_mlt_train_Hindi, \
    make_mlt_train_chlat, make_mlt_train_jp, \
    make_mlt_train_bangla, make_mlt_train_Arabic
from neko_sdk.ocr_modules.lmdbcvt.mltlike import make_rctw_train  # well thay look alike :-)
from osocr_tasks.tasksg1.ch_jpn_osocr.dict3817MCmetafy import make_chlat_mc_dict
from osocr_tasks.tasksg1.ch_jpn_osocr.dict3817SCmetafy import make_chlat_sc_dict
from osocr_tasks.tasksg1.ch_jpn_osocr.dict3817WTmetafy import make_chlat_wt_dict
from osocr_tasks.tasksg1.ch_jpn_osocr.dictchslatkrMCmetafy import make_chlatkr_mc_dict
from osocr_tasks.tasksg1.dscs import get_ds;
from osocr_tasks.tasksg1.dscs import makept


class neko_osocr_datamaker:
    def __init__(this,sroot,croot,droot):
        this.paths={
            "artroot":sroot+"/art",
            "ctwtrgtroot":sroot+"/ctw/gtar/train.jsonl",
            "ctwtrimroot": sroot+"/ctw/itar",
            "mltroot": sroot+"/mlt/real",
            "mltsynthchroot":sroot+"/mlt/Chinese",
            "rctwtrroot": sroot+"/rctw_train/train",
            "lsvttrjson":sroot+"/lsvt/train_full_labels.json",
            "lsvttrimgs":sroot+"/lsvt/imgs/",
            "dict_root": droot+"dicts",
            "desroot":droot,
            "cacheroot":croot,
            "fntpath": [sroot + "fonts/NotoSansCJK-Regular.ttc"],
            "bafntpath":[sroot + "fonts/NotoSansBengali-Regular.ttf"],
            "hndfnt":[sroot + "fonts/NotoSansDevanagari-Regular.ttf"],
            "arfntpath":[sroot + "fonts/NotoSansArabic-Regular.ttf"]
        }

    def make_kr(this):
        rawmltkr = os.path.join(this.paths["cacheroot"], "mlttrkr");
        # skr = os.path.join(this.paths["desroot"], "mltkrdb_seen");
        hkreval = os.path.join(this.paths["desroot"], "mlttrkr_hori");
        make_mlt_train_Korea(this.paths["mltroot"],
                       rawmltkr);
        shfilter(rawmltkr, latin62.union(t1_3755).union(korean), hkreval);
        makept(hkreval,
               this.paths["fntpath"],
               os.path.join(this.paths["dict_root"], "dabkrmlt.pt"),
               latin62,
               symbol.union(t1_3755)
               );
        # If you want to train model with KR.... But this another protocol.
        trdskrpath=os.path.join(this.paths["dict_root"], "dabclkMC.pt");
        make_chlatkr_mc_dict(this.paths["fntpath"],
                             trdskrpath);
        pass;
    def make_hn(this):
        raw_evalh = os.path.join(this.paths["cacheroot"], "mlttrhindi")
        hevalh = os.path.join(this.paths["desroot"], "mlttrhn_hori")
        make_mlt_train_Hindi(this.paths["mltroot"],
                       raw_evalh);

        hindlatchars=list(set(get_ds(raw_evalh)).difference(korean.union(symbol)));
        shfilter(raw_evalh,hindlatchars,hevalh);

        makept(hevalh,
               this.paths["hndfnt"],
               os.path.join(this.paths["dict_root"], "dabhnmlt.pt"),
               latin62,
               symbol.union(korean)
               )

    def make_bear_lang(this): # kuma kuma kuma bear (x
        # Bengali
        rawbangla = os.path.join(this.paths["cacheroot"], "mlttrbengala");
        hevalb=os.path.join(this.paths["desroot"], "mlttrbe_hori")

        make_mlt_train_bangla(this.paths["mltroot"],rawbangla);
        bcs = harvast_cs(rawbangla);
        bcs=bcs.difference(symbol);
        shfilter(rawbangla,bcs,hevalb);

        # Arab
        rawarab = os.path.join(this.paths["cacheroot"], "mlttrarab");
        hevalar = os.path.join(this.paths["desroot"], "mltardb_hori");
        make_mlt_train_Arabic(this.paths["mltroot"],rawarab);
        bcs = harvast_cs(rawarab);
        bcs=bcs.difference(symbol);
        shfilter(rawarab,bcs,hevalar);

        makept(rawarab,
               this.paths["arfntpath"],
               os.path.join(this.paths["dict_root"], "dabbemlt.pt"),
               latin62,
               symbol.union(korean)
               )


    def make_jpn(this):
        raw_eval = os.path.join(this.paths["cacheroot"], "mlttrjp")
        heval=os.path.join(this.paths["desroot"], "mlttrjp_hori");

        make_mlt_train_jp(this.paths["mltroot"],
                       raw_eval);

        jplatchars=list(set(get_ds(raw_eval)).difference(korean.union(symbol)));
        # # Like we said, we do not handle vertical scripts(It breaks batching and adds more effort on coding to transpose them. )
        shfilter(raw_eval,jplatchars,heval);

        makept(heval,
               this.paths["fntpath"],
               os.path.join(this.paths["dict_root"], "dabjpmlt.pt"),
               latin62,
               symbol.union(korean) # To black list symbols and korean.
               )


    def make_chlat_training(this):
        rawart = os.path.join(this.paths["cacheroot"], "artdb");
        rawrctwtr = os.path.join(this.paths["cacheroot"], "rctwtrdb");
        rawlsvt = os.path.join(this.paths["cacheroot"], "lsvtdb");
        rawmltchlat = os.path.join(this.paths["cacheroot"], "mlttrchlat");
        rawctw = os.path.join(this.paths["cacheroot"], "ctwdb");

        sart = os.path.join(this.paths["desroot"], "artdb_seen");
        srctwtr = os.path.join(this.paths["desroot"], "rctwtrdb_seen");
        slsvt = os.path.join(this.paths["desroot"], "lsvtdb_seen");
        smltchlat = os.path.join(this.paths["desroot"], "mlttrchlat_seen");
        sctw = os.path.join(this.paths["desroot"], "ctwdb_seen");

        rawtr=[os.path.join(rawart,"train"),rawrctwtr,rawlsvt,rawmltchlat,rawctw];
        fintr=[sart,srctwtr,slsvt,smltchlat,sctw];



        make_art_lmdb(this.paths["artroot"],
                      rawart);

        make_rctw_train(this.paths["rctwtrroot"],
                        rawrctwtr);

        make_ctw(this.paths["ctwtrgtroot"],
                 this.paths["ctwtrimroot"],
                 rawctw);

        makelsvt(this.paths["lsvttrjson"],
                 this.paths["lsvttrimgs"],
                 rawlsvt);
        make_mlt_train_chlat(
            this.paths["mltroot"],
            rawmltchlat
        );

        for s, d in zip(rawtr, fintr):
            shfilter(s, latin62.union(t1_3755), d);

        # Label transductive
        make_chlat_wt_dict(this.paths["fntpath"], os.path.join(this.paths["dict_root"], "dab3791WT.pt"));

        # Inductive
        make_chlat_sc_dict(this.paths["fntpath"],os.path.join(this.paths["dict_root"],"dab3791SC.pt"));
        make_chlat_mc_dict(this.paths["fntpath"], os.path.join(this.paths["dict_root"],"dab3791MC.pt"));

        # For OSOCR and VSDF, we used transductive setup, but the performance will not drop if you opt to the inductive setup
        # The models are not utilizing the extra information in a beneficial way.
        # For OSOCR, the performances are on par of the Label transductive ver,
        # For VSDF you gets better performance trained with inductive dict....


    def make_all(this):

        # #
        # #the mlt annotation is a little bit messy, some korean scripts are mixed in
        #

        os.makedirs(this.paths["dict_root"],exist_ok=True);
        #

        this.make_chlat_training();
        this.make_jpn();
        this.make_kr();
        this.make_hn();
        this.make_bear_lang();

    # removing all vertical clips in training set and clips with unseen characters.
    # These are not what aims to solve in this paper.

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        SROOT = sys.argv[1]
        CROOT = sys.argv[2]
        DROOT = sys.argv[3]
    else:
        SROOT = "/run/media/lasercat/writebuffer/deploy/"
        CROOT = "/run/media/lasercat/cache2/"
        DROOT="/run/media/lasercat/writebuffer/cachededlmdbs/"
    neko_osocr_datamaker(SROOT,CROOT,DROOT).make_all();
