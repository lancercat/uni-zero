from osocrNG.lsctNG.libtrdg.functional import gen_bg,gen_bg_in_parallel
from multiprocess.pool import Pool
import numpy as np
class bg_gen:
    def __init__(this):
        this.pool=Pool(16);
    def get_bg(this,background_type, background_height, background_width, background_img):
        return gen_bg(background_type, background_height, background_width, background_img);
    def gen_bg_in_parallel(this,params):
        return gen_bg_in_parallel(params,this.pool);


