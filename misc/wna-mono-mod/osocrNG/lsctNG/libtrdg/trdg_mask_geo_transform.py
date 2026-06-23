import os
import random as rnd

import numpy as np;
from PIL import Image, ImageFilter, ImageColor

from osocrNG.lsctNG.libtrdg import distort_mask_only,distort_mask_list
from osocrNG.lsctNG.libtrdg.trdg_core import generate
class trdg_mask_geo_transforms:
    @classmethod
    def transform(cls,
                  mask, margins,
                  size,
                  skewing_angle,
                  random_skew,
                  distorsion_type,
                  distorsion_orientation,
                  orientation):
        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
             distorted_mask = distort_mask_only.sin(
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_mask = distort_mask_only.cos(
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_mask = distort_mask_only.random(
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_mask.size[0]
                * (float(size - vertical_margin) / float(distorted_mask.size[1]))
            )

            resized_mask = distorted_mask.resize((new_width, size - vertical_margin))

        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_mask.size[1])
                * (float(size - horizontal_margin) / float(distorted_mask.size[0]))
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.LANCZOS
            )

        else:
            raise ValueError("Invalid orientation")

        return resized_mask
    @classmethod
    def transform_stub(cls,params):
        return cls.transform(*params);
    @classmethod
    def transform_parallel(cls,params,pool):
        return pool.map(cls.transform_stub,params);
    @classmethod
    def transform_stub_to_np(cls,params):
        if(params[0] is None):
            return None;
        return np.array(cls.transform_stub(params))[:,:,0:1]

    @classmethod
    def transform_parallel_to_np(cls,params,pool):
        return pool.map(cls.transform_stub_to_np,params);
class trdg_mask_geo_transforms_insmask:
    @classmethod
    def transform(cls,
                  mask,insmask, margins,
                  size,
                  skewing_angle,
                  random_skew,
                  distorsion_type,
                  distorsion_orientation,
                  orientation):
        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )
        rotated_insmasks=[m.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        ) for m in insmask];

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_mask = rotated_mask;
            disorted_ins_masks=rotated_insmasks;
        elif distorsion_type == 1:
             distorted_mask,disorted_ins_masks = distort_mask_list.sin(
                rotated_mask,rotated_insmasks,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_mask,disorted_ins_masks = distort_mask_list.cos(
                rotated_mask,rotated_insmasks,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_mask,disorted_ins_masks = distort_mask_list.random(
                rotated_mask,rotated_insmasks,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_mask.size[0]
                * (float(size - vertical_margin) / float(distorted_mask.size[1]))
            )
            new_width=max(new_width,1);
            new_height=max(size - vertical_margin,1)
            resized_mask = distorted_mask.resize((new_width,new_height ))
            resized_insmsks=[m.resize((new_width,new_height )) for m in disorted_ins_masks]

        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_mask.size[1])
                * (float(size - horizontal_margin) / float(distorted_mask.size[0]))
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.LANCZOS
            )
            resized_insmsks=[m.resize(
                (size - horizontal_margin, new_height), Image.LANCZOS
            ) for m in disorted_ins_masks]

        else:
            raise ValueError("Invalid orientation")

        return resized_mask, resized_insmsks

