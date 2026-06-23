from osocrNG.sptokens import tEOS,tBLANC
from neko_sdk.neko_framework_NG.libprototyping.modules.embedding_zombie_mk2 import neko_embedding_zombie_normed_mk2
class neko_quick_and_dirty_sp(neko_embedding_zombie_normed_mk2):
    DFT_capacity=5;

    def __init__(this,param):
        super().__init__(param);
        this.register_utfs_as_pk([tBLANC,tEOS]);

    def forward(this):
        utf_list= this.utf_list;
        protos=this.core[:len(this.utf_list)];
        return protos,utf_list;
if __name__ == '__main__':
    sph=neko_quick_and_dirty_sp({});
    a=sph();
    pass;
