def neko_get_arg(key,args,default=None):
    if(default is None):
        v=args[key];
    else:
        v=default;
    if(key in args):
        v=args[key]
    if (v=="NEP_skipped_NEP"):
        v =None;
    elif(v=="NEP_default_NEP"):
        v=default;
    return v;

def neko_set_arg_if_not_already(key,args,value):
    if(key not in args):
        args[key]=value;

def neko_get_arg_dict(args,default_dict=None):
    ret_args={}
    for k in default_dict:
        if(k not in args):
            ret_args[k]=default_dict[k];
        else:
            ret_args[k]=default_dict[k];
    return args

def neko_get_defarg(key,args,pfx="_name"):
    return neko_get_arg(key+pfx,args,key);
def neko_get_set_arg(key,args,default=None):
    if(default is None):
        v=args[key];
    else:
        v=default;
    if(key in args):
        v=args[key]
    if (v=="NEP_skipped_NEP"):
        v =None;
    else:
        args[key]=v;
    return v;
