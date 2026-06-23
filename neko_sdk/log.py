import sys
import time
from neko_sdk.die import die

def err(text):
    print(text);
def warn(*args):
    print("WARN:",*[str(i) for i in args]);
    sys.stdout.flush();
    sys.stderr.flush();
def info(*args):
    print("INFO:",*[str(i) for i in args]);
    sys.stdout.flush();
    sys.stderr.flush();
def fatal(*args):
    print("FATAL:",*[str(i) for i in args]);
    sys.stdout.flush();
    sys.stderr.flush();
    time.sleep(1);
    die(9);


def collapse_tree(data, sep="-"):
    # If not a dict, we've reached a leaf value
    if not isinstance(data, dict):
        return data

    # Process all children first (bottom-up approach)
    new_dict = {k: collapse_tree(v, sep) for k, v in data.items()}

    # Check if this level needs collapsing
    if len(new_dict) == 1:
        key = list(new_dict.keys())[0]
        val = new_dict[key]

        # Only collapse if the child is also a dictionary
        if isinstance(val, dict):
            # Return a new dict with the merged key
            collapsed_key = f"{key}{sep}{list(val.keys())[0]}"
            return {collapsed_key: val[list(val.keys())[0]]}

    return new_dict
def tree_view(dic):
    tree = {};
    for k in dic:
        nodes = k.split("-");
        stree = tree
        for n in nodes:
            if (n not in stree):
                stree[n] = {};
            stree = stree[n];
        stree[k] = dic[k];
    return tree;
def prefixed(tv,prfx_lst):
    if(len(prfx_lst)==0):
        return tv;
    if(prfx_lst[0] in tv):
        return prefixed(tv[prfx_lst[0]],prfx_lst[1:]);
    return tv;
def tree_view_prefixed(dic,prfx):
    prfx_lst=prfx.split("-");
    tv=tree_view(dic);
    return prefixed(tv,prfx_lst);