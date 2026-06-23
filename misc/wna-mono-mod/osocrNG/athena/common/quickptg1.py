import os
import shutil

import regex

from osocr_tasks.tasksg1.dscs import makept


def prepare_pt(langroot):
    dictp=os.path.join(langroot,"meta","dict.pt");
    if(os.path.exists(dictp)):
        print("skipping pt build, to force rebuilding, remove", dictp);
        return dictp;

    with open(os.path.join(langroot,"meta","alphabets.txt"))as fp:
        chars=[l.strip() for l in fp];
        allch=[];
        masters=[];
        servants=[];
        for ch in chars:
            if(len(ch)):
                l_= regex.findall(r'\X', ch, regex.U);
                l=[];
                for i in l_:
                    if(i!=" "):
                        l.append(i); # Someone may accidentally separate characters with space....
                allch+=l;
                for i in range(1,len(l)):
                    masters.append(l[0]);
                    servants.append(l[i]);
    allch=list(set(allch));
    fntp=os.path.join(langroot,"meta","notofont.ttf");
    makept(None, [fntp],
           dictp,
           allch, {}, masters=masters, servants=servants);
    return dictp;


# setup NG proto (basically just copy it as vismeta)
# zero-shot semantic meta will come after BertAlive project.
# Well I am but a cat so I can't fork myself.
def prepare_pt_ng(langroot):
    dictp=prepare_pt(langroot);
    v2dictr=os.path.join(langroot, "dict");
    os.makedirs(v2dictr, exist_ok=True);
    v2dictp = os.path.join(langroot, "dict", "vismeta.pt");
    shutil.copy(dictp,v2dictp);
    return dictp,v2dictr;
