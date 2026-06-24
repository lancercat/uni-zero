import copy
import os.path
import unicodedata
from itertools import chain

import torch;
import tqdm
from fontTools.ttLib import TTFont
from neko_sdk.log import warn
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path
class fntmgmt:
    NORMAL_CHARACTER=["Lo","Lu","Ll","So","Sm","Nd","Nl","No",]
    @classmethod
    def parse_sub_4(cls,sub):
        ret=[];
        for k in sub.ligatures.keys():
            for l in sub.ligatures[k]:
                magic=[k]+l.Component;
                ret.append(magic);
        return ret;
    @classmethod
    def parse_sub_1(cls,sub):
        return sub.mapping.keys();

    @classmethod
    def parse_sub_2(cls, sub):
        return sub.mapping.values();

    @classmethod
    def parse_sub(cls,sub):
        if(sub.LookupType==1):
            return cls.parse_sub_1(sub);
        elif(sub.LookupType==2):
            return cls.parse_sub_2(sub);
        elif(sub.LookupType==3):
            pass;

        elif(sub.LookupType==4):
            return cls.parse_sub_4(sub);
        else:
            # print(sub.LookupType);
            return [];
    @classmethod
    def get_charset(cls, fp):
        ttf = TTFont(fp, 0,fontNumber=0, verbose=0, allowVID=0,
                     ignoreDecompileErrors=True)
        chars = chain.from_iterable([chr(y[0]) for y in x.cmap.items()] for x in ttf["cmap"].tables)
        return set(chars);

    # supports gsub. With respect, this is bloodily  over-complex
    @classmethod
    def get_charset_gen2(cls, fp):



        ttf = TTFont(fp, 0, fontNumber=0, verbose=0, allowVID=0,
                     ignoreDecompileErrors=True)
        dic={};
        rdic={}
        def handy(y):
            dic[y[1]]=chr(y[0]);
            rdic[chr(y[0])]=dic[y[1]];
            return chr(y[0])
        schars_ = list(chain.from_iterable([handy(y) for y in x.cmap.items()] for x in ttf["cmap"].tables))
        schars=[];
        for c in schars_:
            if unicodedata.category(c) in cls.NORMAL_CHARACTER:
                schars.append(c);
        cchars=[];
        flag=True;
        try:
            _=ttf["GSUB"].table.LookupList;
        except:
            flag=False;
        if(flag and "GSUB" in ttf and ttf["GSUB"].table.LookupList is not None):
            for sub in ttf["GSUB"].table.LookupList.Lookup:
                for t in sub.SubTable:
                    mag=cls.parse_sub(t);
                    if(mag is None ):
                        continue;

                    codes=list(mag);
                    for c in codes:
                        try:
                            if(type(c)==list):
                                rc="".join(dic[c_.split(".")[0].split("_")[0]] for c_ in c);
                            else:
                                rc=dic[c];
                            cchars.append(rc);
                        except:
                            # print(c);
                            for c_ in c:
                                if(c_.split(".")[0].split("_")[0] not in dic):
                                    pass;
                                    # print("what is ",c_);
        return set(cchars),set(schars);

    @classmethod
    def get_charset_gen3(cls, fp):

        ttf = TTFont(fp, 0, fontNumber=0, verbose=0, allowVID=0,
                     ignoreDecompileErrors=True)
        dic = {};
        rdic = {}

        def handy(y):
            dic[y[1]] = chr(y[0]);
            rdic[chr(y[0])] = dic[y[1]];
            return chr(y[0])

        schars_ = list(chain.from_iterable([handy(y) for y in x.cmap.items()] for x in ttf["cmap"].tables))
        schars = [];
        for c in schars_:
            if unicodedata.category(c) in cls.NORMAL_CHARACTER:
                schars.append(c);
        cchars = [];
        flag = True;
        try:
            _ = ttf["GSUB"].table.LookupList;
        except:
            flag = False;
        if (flag and "GSUB" in ttf and ttf["GSUB"].table.LookupList is not None):
            for sub in ttf["GSUB"].table.LookupList.Lookup:
                for t in sub.SubTable:
                    mag = cls.parse_sub(t);
                    if (mag is None):
                        continue;

                    codes = list(mag);
                    for c in codes:
                        try:
                            if (type(c) == list):
                                rc = "".join(dic[c_.split(".")[0].split("_")[0]] for c_ in c);
                            else:
                                rc = dic[c];
                            cchars.append(rc);
                        except:
                            # print(c);
                            for c_ in c:
                                if (c_.split(".")[0].split("_")[0] not in dic):
                                    pass;
                                    # print("what is ",c_);
        return set(cchars), set(schars), set(schars_).union(set(cchars));
    @classmethod
    
    def init_charset(this,fnt_d):
        for k in fnt_d:
            this.charset_d[k]=this.get_charset(fnt_d[k]);
            torch.save(this.charset_d,"charset_d.pt")
            print(k,"scanned");
    def load_charset(this):
        this.charset_d=torch.load("charset_d.pt");
    def __init__(this,fnt_d):
        this.charset_d={};
        if(fnt_d is None):
            this.load_charset();
        else:
            this.init_charset(fnt_d);
        pass;

from neko_sdk.environment.root import find_data_root
class fntmgr:
    KEY_fnt_charset="fnt_charset";
    KEY_fnt_grp = "fnt_grp";
    KEY_spaces = "spaces"; # spaces characters used in this language
    KEY_providers="providers"
    KEY_name_2_id="name_2_id"; # simply we can't maintain a path list for each character. will be huge.
    KEY_id_2_name="id_2_name";
    DFT_key="fntidx.pt";
    BAD_APPLES=["adobeblank"];
    def empty(this):
        this.meta={}
        this.meta[this.KEY_fnt_charset] = {};
        this.meta[this.KEY_fnt_grp]={};
        this.meta[this.KEY_spaces]={};
        this.meta[this.KEY_providers]={};
        this.meta[this.KEY_name_2_id]={};
        this.meta[this.KEY_id_2_name] = [];

    def load(this,path):
        this.meta=torch.load(path,weights_only=False);
    @classmethod
    def printable(cls,tokens, charset):
        r=[];
        for t in tokens:
            flag=True;
            for c in t:
                if(c not in charset):
                    flag=False;
                    break;
            if flag:
                r.append(t);
        return r;
    def tryload(this,idxpath,fntroot=None,size=32):
        try:
            this.load(idxpath);
        except:
            this.one_grp_bootstrap(os.path.join(fntroot));
            this.save(idxpath);
    def save(this,path):
        torch.save(this.meta,path)
    def __init__(this):
        this.empty();

    def addgrp(this,name,space):
        this.meta[this.KEY_fnt_grp][name]=[];
        this.meta[this.KEY_spaces][name] = space;
    def extract_cs(this,fnt):
        return  list(fntmgmt.get_charset(fnt));
    def get_cs(this,name):
        return this.meta[this.KEY_fnt_charset][name];
    def fonts_available(this,string):
        if(len(string) ==0):
            return set();
        s=copy.copy(this.meta[this.KEY_providers][string[0]]) ;
        for c in string[1:]:
            s=s.intersection(this.meta[this.KEY_providers][c]);
        return [this.meta[this.KEY_id_2_name][fid] for fid in s]




    def add_fnt(this,folder,fnt,grp_name):
        assert (fnt not in this.meta[this.KEY_fnt_charset]);
        try:
            ImageFont.truetype(font=fnt, size=32)
        except:
            warn(fnt, "won't render");
            return 9
        for rf in this.BAD_APPLES:
            if(fnt.find(rf)!=-1):
                warn("known bad apple", fnt, "skipped");
                return 9;
        try:
            fnt_cs = this.extract_cs(os.path.join(folder,fnt));
            fid=len(this.meta[this.KEY_id_2_name]);
            if(len(fnt_cs)==0):
                warn("empty font", fnt, "skipped");
                return 9;

        except:
            warn("bad font",fnt,"skipped");
            return 9;

        for c in fnt_cs:
            if(c not in  this.meta[this.KEY_providers]):
                this.meta[this.KEY_providers][c]=set();
            this.meta[this.KEY_providers][c].add(fid);
        fnt=str(Path(fnt).relative_to(folder));
        this.meta[this.KEY_fnt_grp][grp_name].append(str(fnt));
        this.meta[this.KEY_name_2_id][fnt]=fid;
        this.meta[this.KEY_id_2_name].append(fnt)
        this.meta[this.KEY_fnt_charset][fnt]=fnt_cs;
        return 0;


    def one_grp_bootstrap(this,folder):
        this.addgrp("main"," ");

        with open(os.path.join(folder,"all.txt"),"r") as fnts:
            afs= [f.strip() for f in fnts];
            for f in tqdm.tqdm(afs):
                this.add_fnt(folder,f,grp_name="main");


class make_meta(fntmgr):
    KEY_segment_length = "fnt_charset" # max length of language of this font group ;
    KEY_grpseg = "grpseg"; # typical segment in a line

    def addgrp(this,name,space,segment_cnt,segment_length):
        super().addgrp(name,space);
        this.meta[this.KEY_grpseg][name] = segment_cnt;
        this.meta[this.KEY_segment_length][name] = segment_length;

        this.meta[this.KEY_fnt_grp][name]=[];
        this.meta[this.KEY_spaces][name] = space;

    def __init__(this,all_chars,blacklisted_chars):
        super().__init__();
        this.meta[this.KEY_grpseg] = {};
        this.meta[this.KEY_segment_length] = {};
        this.valid=set(all_chars)-set(blacklisted_chars);
    def extract_cs(this,fnt):
        fnt_cs=fntmgmt.get_charset(fnt);
        return this.printable(this.valid,fnt_cs);


if __name__ == '__main__':
    mgr=fntmgr();
    mgr.tryload("/home/lasercat/ssddata/synth_lsct/cache/finfo251105.pt",
                "/home/lasercat/ssddata/synth_lsct/");
    a=mgr.fonts_available("개Сука犬狗doge");
    pass;
