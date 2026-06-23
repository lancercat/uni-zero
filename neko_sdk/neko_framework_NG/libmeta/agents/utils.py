def mk_id_dict(tdict):
    rd = {};
    bls=set()
    for c in tdict:
        if (type(c) == int):
            rd[c] = [tdict[c]];# make pk first, nice to have but don't make assumptions
            bls.add(c);
            bls.add(tdict[c]);
    for c in tdict:
        if (c not in bls):
            rd[tdict[c]].append(c); # make rest
    return rd;
def mk_meta(utf_list,master_dict):
    tdict = {};
    i = 0;
    for u in utf_list:
        if (u in tdict):
            continue;
        if(u in master_dict):
            pk=master_dict[u];
        else:
            pk = u;
        if(pk not in tdict):
            tdict[i]=pk;
            tdict[pk]=i;
            i+=1;
        tdict[u]=tdict[pk];
    return tdict;
def sorted_utf(tdict):
    sutf_list = [];
    idd = mk_id_dict(tdict);
    for k in sorted(list(idd.keys())):
        sutf_list += idd[k];
    return sutf_list;