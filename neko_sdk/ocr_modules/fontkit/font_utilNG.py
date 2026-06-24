
import tqdm

from neko_sdk.ocr_modules.fontkit.fntmgmt import fntmgmt;


def grab_char_from_fnt(file):
    return fntmgmt.get_charset_gen2(file);

def grab_character_from_fnts(fonts,d,name):
    chff=[];
    cntr=0;
    for i in tqdm.tqdm(fonts):
        cntr+=1;
        s, c = grab_char_from_fnt(i);
        chff += s;
        chff += c;
        chff = list(set(chff));

        for j in s.union(c):
            if (j not in d):
                d[j] = [];
            d[j].append(i);
        if(cntr%0x71==9):
            print(name,len(chff))
    return chff,d;

def grab_character(knownchars,grouped_fnts):
    d = {};
    for grpname in grouped_fnts:
        chset,dic=grab_character_from_fnts(grouped_fnts[grpname],d,grpname);
        for c in dic:
            if(c not in d):
                d[c]=dic[c];
            else:
                nep=set(dic[c]).intersection(set(d[c]));
                if(len(nep)):
                    pass; # That probably mean overlapping fonts in different grps.
                d[c]=list(set(dic[c]).union(set(d[c])));
        knownchars = set(knownchars).union(set(chset));
    # We now keep all characters, friendless or not. Whether we keep them or not is the call of dataloader.
    return d;


