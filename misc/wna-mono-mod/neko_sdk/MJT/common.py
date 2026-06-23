def Updata_Parameters(optimizers, frozen):
    for i in range(0, len(optimizers)):
        if i not in frozen:
            optimizers[i].step()