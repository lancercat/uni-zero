import glob
import json;
import os;
import shutil;

import cv2;

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper;

BRESSAY_LST=[("##@@???@@##","①"),
("$$@@???@@$$","②"),
("--xxx--","③"),

("@@???@@","④"),
("##--","⑤"),
("--##","⑥"),
("$$--","⑦"),
("--$$","⑧"),
("##","⑩"),
("--","⑪"),
];
BRESSAY_PATCHS=[
["porâneo sabe apenas, somar, sem nunca se atentar ao resultado. --disso-- ##Assim# não percebemos que","porâneo sabe apenas, somar, sem nunca se atentar ao resultado. --disso-- ##Assim## não percebemos que"],
["acesso a tratamento de doenças mentais ao criar o Sistema Único de Saúde (SUS). Entretanto, --xxx-","acesso a tratamento de doenças mentais ao criar o Sistema Único de Saúde (SUS). Entretanto, --xxx--"],
["tes seja benéfico ao doente. Dessa forma, uma sociedade @@grolada@@ em um governo omisso","tes seja benéfico ao doente. Dessa forma, uma sociedade grolada em um governo omisso"],
["aplicativos e assim, --(xxx)-- causando o sentimento de exclusão social dos seres com doenças mentais.","aplicativos e assim, --xxx-- causando o sentimento de exclusão social dos seres com doenças mentais."],
["te\", é --xxx-- @@cupladora@@ de uma sociedade que não dá a devida importância a sintomas que podem culmi-","te\", é --xxx-- cupladora de uma sociedade que não dá a devida importância a sintomas que podem culmi-"],
["ceos dos transtornos, --estes-- os indivíduos perdem a capacidade de interagir socialmente, @@frián@@","ceos dos transtornos, --estes-- os indivíduos perdem a capacidade de interagir socialmente, frián"],
["constante entre brancos e escravizados inserido em uma lógica autoritária. Apesar de --(xxx)--","constante entre brancos e escravizados inserido em uma lógica autoritária. Apesar de --xxx--"],
["é um entrave para a resolução do --(xxx)-- problema. Durante sua his-","é um entrave para a resolução do --xxx-- problema. Durante sua his-"],
["em --(xxx)-- 1988, resguardem o direito à cidadania e ao lazer, a realidade choca, frente à pe-","em --xxx-- 1988, resguardem o direito à cidadania e ao lazer, a realidade choca, frente à pe-"],
["Claro e os amigos por terem medo de serem rejeitados. Em suma, #os# pensamentos errônios da popula-","Claro e os amigos por terem medo de serem rejeitados. Em suma, ##os## pensamentos errônios da popula-"],
["#menstun,# -mención- em seu livro \" O cidadão de Papel\" a motoria das leis navonals amparam os cidadãos somen","##menstun,## --mención-- em seu livro \" O cidadão de Papel\" a motoria das leis navonals amparam os cidadãos somen"],
["nováves a #estarem# estaremos em desequilíbrio emocional anem há o reforço na complicação e estigmas contro a minoria.","nováves a ##estarem## estaremos em desequilíbrio emocional anem há o reforço na complicação e estigmas contro a minoria."],
["lução dos enfermidades mentas, a fim de desconstruir #impasem# -estole- depreciativo que geram a exclusão dos enfermos.","lução dos enfermidades mentas, a fim de desconstruir ##impasem## --estole-- depreciativo que geram a exclusão dos enfermos."],
["riamente o é para outro. Consoante ao artigo publicado --xxx-- ##xxx## Leonardo Lichote so-","riamente o é para outro. Consoante ao artigo publicado --xxx-- ##@@???@@## Leonardo Lichote so-"],
["uso contestável como simples produto, ela não piorou, apenas ##xxx## transformou--(se)--, de","uso contestável como simples produto, ela não piorou, apenas ##@@???@@## transformou—se--, de"],
["vernos Estaduais e -xxx-- Municipais, deve catalogar as áreas com maior necessidade de","vernos Estaduais e --xxx-- Municipais, deve catalogar as áreas com maior necessidade de"],
["entender como as atitudes equivocadas do povo e do governo nacional agravam esse -xxx--","entender como as atitudes equivocadas do povo e do governo nacional agravam esse --xxx--"],
["Logo, a constante exposição dos jovens ao fenômeno das \"fakes news\", no mundo digital, desencadeia -xxx-- uma descrença em Re-","Logo, a constante exposição dos jovens ao fenômeno das \"fakes news\", no mundo digital, desencadeia --xxx-- uma descrença em Re-"],
["Estado reduz sua participação de forma que --por-- serviços como educação e saúde, @@??@@ oferecidos pelo Es-","Estado reduz sua participação de forma que --por--  serviços como educação e saúde, @@???@@ oferecidos pelo Es-"],
["menor poder aquisitivo ⑩@@@⑩","menor poder aquisitivo --xxx--"],
["Brasil finalmente poderá, assim, remover esse estigma que @@maria@@ parte da --xxx-- população.","Brasil finalmente poderá, assim, remover esse estigma que maria parte da --xxx-- população."],
]


def handle_bressy_string(gt,fix=True):
    if(fix):
        for i in BRESSAY_PATCHS:
            gt=gt.replace(i[0],i[1]);
    for i in BRESSAY_LST:
        gt=gt.replace(i[0],i[1]);

    return gt;



def make_bressy_lmdb(root,dst,type,split):
    sdst = os.path.join(dst,type, split);
    shutil.rmtree(sdst, True);
    os.makedirs(sdst,exist_ok=True);
    db = im_lmdb_wrapper(sdst,1);
    fnlst=[];
    with open(os.path.join(root,"sets",split+".txt") )as fp:
        fnlst=[l.strip() for l in fp];
    for fn in fnlst:
        ilist=glob.glob(os.path.join(root,"data",type,fn,"*.png"));
        for i in ilist:
            im = cv2.imread(i);
            txt=i.replace("png","txt");
            lns=[];
            with open(txt,"r") as fp:
                lns=[l for l in fp];
            gt="".join(lns);
            if(split == "training"):
                fix=True;
            else:
                fix=False;
            ngt=handle_bressy_string(gt,fix);
            if (ngt.find("@") != -1 or ngt.find("#") != -1 or ngt.find("??")!=-1 or ngt.find("xxx")!=-1):
                print(txt,"of",type,fn,split);
                print(ngt);
                print(gt);
            lang="Portuguese";
            db.add_data_utf(im, gt, lang);


    db.end_this();


if __name__ == '__main__':
    splits=["test","training","validation"]
    types=["lines","paragraphs","pages"];
    for t in types:
        for s in splits:
            make_bressy_lmdb("/run/media/lasercat/thirdeye/bressy/bressay/","/home/lasercat/ssddata/bressay",t,s);