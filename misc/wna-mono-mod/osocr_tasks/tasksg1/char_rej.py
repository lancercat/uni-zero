import torch

from neko_sdk.ocr_modules.renderlite.addfffh import refactor_meta, add_masters, finalize


def buildrej(srcpt,dstpt,k=400):
   a=torch.load(srcpt);
   dst={};
   dst["chars"]=a["chars"][:k];
   dst["protos"]=a["protos"][:k+1]+[None];
   dst["sp_tokens"]=a["sp_tokens"];
   dst["label_dict"]={};
   cnt=0;
   for c in dst["sp_tokens"]:
      dst["label_dict"][c]=cnt;
      cnt+=1;
   for c in dst["chars"]:
      dst["label_dict"][c]=cnt;
      cnt+=1;
   dst["label_dict"]['[UNK]']=cnt;
   refactor_meta(dst);
   add_masters(dst, [], []);
   dst = finalize(dst);
   torch.save(dst,dstpt);
   print("built",dstpt)
if __name__ == '__main__':
    import sys
    if(len(sys.argv)>1):
        DROOT = sys.argv[1];
    else:
        DROOT= "/run/media/lasercat/cache2/"

    HWDB_ROOT = DROOT+"/HWDB/pami_ch_fsl_hwdb/hwdbfsl_10_1/cuwu_evalhwdbfsl_10_1/";
    CTW_ROOT = DROOT+"/run/media/lasercat/cache2/ctwch/ctwfsl_5_1eval/";

    # buildrej(HWDB_ROOT+"dict.pt",HWDB_ROOT+"dictrej.pt");
    buildrej(HWDB_ROOT+"dict.pt",
             HWDB_ROOT+"dictrej500.pt",k=500);
    buildrej(HWDB_ROOT+"dict.pt",
             HWDB_ROOT+"dictrej400.pt",k=400);
    buildrej(HWDB_ROOT+"dict.pt",
             HWDB_ROOT+"dictrej200.pt",k=200);
    buildrej(HWDB_ROOT+"dict.pt",
             HWDB_ROOT+"dictrej100.pt",k=100);

    buildrej(CTW_ROOT+"dict.pt",CTW_ROOT+"dictrej250.pt",k=250);
    buildrej(CTW_ROOT+"dict.pt",CTW_ROOT+"dictrej200.pt",k=200);
    buildrej(CTW_ROOT+"dict.pt",CTW_ROOT+"dictrej100.pt",k=100);
    buildrej(CTW_ROOT+"dict.pt",CTW_ROOT+"dictrej50.pt",k=50);
