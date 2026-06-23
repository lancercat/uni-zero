import torch

from osocrNG.charset.chs_cset import t1_3755, cpx3755
from osocrNG.charset.etc_cset import latin62, symbol
from osocrNG.charset.jpn2trd import kanji2cpx;
from osocrNG.charset.jpn_cset import Kyoiku_kanji, hira, kata

Japcommon=kata.union(hira).union(Kyoiku_kanji).union(latin62);

a=torch.load("/media/lasercat/backup/deployedlmdbs/dicts/dabjpmlt.pt");
a["chars"];

trunseen=set(a["chars"]).difference(symbol).difference(t1_3755.union(cpx3755)).difference(hira).difference(kata).difference(latin62);
rare=set(a["chars"]).difference(symbol).difference(Japcommon);
trd={};
for k in kanji2cpx:
    for v in kanji2cpx[k]:
        trd[v]=k;

for k in a["chars"]:
    if(k in trd):
        print(k,"->-",trd[k]);
        if(trd[k] in a["chars"]):
            print("found both in GT",trd[k],k)
            if(trd[k] in t1_3755):
                print( trd[k],"Possible smp chs");
            if(k in cpx3755):
                print(k,"Possible cpx chs");




pass;
