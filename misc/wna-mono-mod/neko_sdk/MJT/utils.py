def update(opt):
    try:
        opt.step();
    except:
        print("Oops",opt);
        opt.step()
        fatal("error");
    return [];


def normgrad(mod):
    mod.normgrad();
    return [];