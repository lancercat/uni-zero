import glob;
import os;

from neko_sdk.lmdbcvt.mlt_patchs import patchs;


class patcher:
    def __init__(this):
        this.unique_map=dict();
        for lang in patchs:
            for c in patchs[lang]:
                this.unique_map[c]=lang;
        this.unique_map_keys=set(this.unique_map.keys());
    def fix_mlt_file(this,file):
        lines=[];
        with open(file,"r") as fp:
            for l in fp:
                sections=l.split(",",9);
                content_cs=set(sections[9]);
                determistic_cs=content_cs.intersection(this.unique_map_keys);
                if(sections[8] == "Mixed" or len(determistic_cs)==0):
                    lines.append(l.strip());
                    continue;
                inferenced_langs=set([this.unique_map[i] for i in determistic_cs]);
                flang=sections[8];
                if (len(inferenced_langs) != 1):
                    flang = "Mixed";
                else:
                    flang=list(inferenced_langs)[0]
                if(sections[8]!=flang):
                    print(sections[9],"was",sections[8],"should be of lang(s)", inferenced_langs);
                    sections[8]=flang;
                l="";
                for i in sections:
                    l+=i+",";
                l=l.strip(',');
                lines.append(l);
        with open(file,"w") as fp:
            fp.writelines(lines);
    def fix_dataset(this,path):
        files=glob.glob(os.path.join(path,"*.txt"));
        for f in files:
            this.fix_mlt_file(f);
        return path;

# patcher().fix_dataset("/home/lasercat/netdata/tmp/imgs")