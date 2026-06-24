from osocrNG.lsctNG.libtrdg.trdg_mask_geo_transform import trdg_mask_geo_transforms,trdg_mask_geo_transforms_insmask
from neko_sdk.cfgtool.argsparse import neko_get_arg
import random

class neko_trdg_mask_deformer:
    PARAM_size="size";
    def set_genpara(this,param):
        this.size=neko_get_arg(this.PARAM_size,param,64);
        this.skewing_angle = [0, 0, 0, 2, 2, 5, 5, 10];
        this.random_skew = [False, False, False, True];
        this.distorsion_type = [False, False, 0, 1, 2];
        this.distorsion_orientation = [0, 1, 2];
        this.alignment = 0;
        this.margins = (5, 5, 5, 5);

    def set_transform(this):
        this.transform=trdg_mask_geo_transforms;

    ### well, rng is not a configurable thing...
    def __init__(this,param,rng=None):
        # meta info
        # background image list
        this.set_genpara(param);
        this.set_transform();
        if(rng is not None):
            this.rng=rng
        else:
            this.rng=random.Random(9);
        pass;
    def drive_param(this,mask,orientation):
        skewing_angle = this.rng.choice(this.skewing_angle);
        random_skew = this.rng.choice(this.random_skew);
        distorsion_type = this.rng.choice(this.distorsion_type);
        distorsion_orientation = this.rng.choice(this.distorsion_orientation);
        margin = this.margins
        size = this.size;
        return (mask,margin,size,
            skewing_angle,
            random_skew,
            distorsion_type,
            distorsion_orientation,
            orientation);

    def drive(this,mask,orientation):
        if mask is None:
            return None;
        skewing_angle = this.rng.choice(this.skewing_angle);
        random_skew = this.rng.choice(this.random_skew);
        distorsion_type = this.rng.choice(this.distorsion_type);
        distorsion_orientation = this.rng.choice(this.distorsion_orientation);
        margin=this.margins
        size = this.size;

        final_mask=this.transform.transform(mask,margin,size,
            skewing_angle,
            random_skew,
            distorsion_type,
            distorsion_orientation,
            orientation
        );
        return final_mask;
    def drive_as_np_inparallel(this,masks,orientations,pool):
        params=[this.drive_param(mask,orientation) for mask,orientation in zip(masks,orientations)];
        return this.transform.transform_parallel_to_np(params,pool);

class neko_trdg_mask_deformer_wins(neko_trdg_mask_deformer):
    def set_transform(this):
        this.transform=trdg_mask_geo_transforms_insmask

    def drive(this,mask,insmask,orientation):
        skewing_angle = this.rng.choice(this.skewing_angle);
        random_skew = this.rng.choice(this.random_skew);
        distorsion_type = this.rng.choice(this.distorsion_type);
        distorsion_orientation = this.rng.choice(this.distorsion_orientation);
        margin=this.margins
        size = this.size;

        final_mask,final_insmask=this.transform.transform(mask,insmask,margin,size,
            skewing_angle,
            random_skew,
            distorsion_type,
            distorsion_orientation,
            orientation
        );
        return final_mask,final_insmask;