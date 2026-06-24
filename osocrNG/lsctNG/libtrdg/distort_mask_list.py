import math
import random as rnd

import numpy as np
from PIL import Image

# distorsion and affine does not necessarily apply to background--- especially if they are random.

def _apply_core(mask, vertical, horizontal, max_offset, func):

    # Nothing to do!
    if not vertical and not horizontal:
        return mask

    # FIXME: From looking at the code I think both are already RGBA
    rgb_mask = mask.convert("RGB")

    mask_arr = np.array(rgb_mask)

    vertical_offsets = [func(i) for i in range(mask_arr.shape[1])]
    horizontal_offsets = [
        func(i)
        for i in range(
            mask_arr.shape[0]
            + (
                (max(vertical_offsets) - min(min(vertical_offsets), 0))
                if vertical
                else 0
            )
        )
    ]

    new_mask_arr = np.zeros(
        (
            # I keep img_arr to maximise the chance of
            # a breakage if img and mask don't match
            mask_arr.shape[0] + (2 * max_offset if vertical else 0),
            mask_arr.shape[1] + (2 * max_offset if horizontal else 0),
            3,
        )
    )

    new_mask_arr_copy = np.copy(new_mask_arr)

    if vertical:
        column_height = mask_arr.shape[0]
        for i, o in enumerate(vertical_offsets):
            column_pos = (i + max_offset) if horizontal else i
            new_mask_arr[
                max_offset + o : column_height + max_offset + o, column_pos, :
            ] = mask_arr[:, i, :]

    if horizontal:
        row_width = mask_arr.shape[1]
        for i, o in enumerate(horizontal_offsets):
            if vertical:
                new_mask_arr_copy[
                    i, max_offset + o : row_width + max_offset + o, :
                ] = new_mask_arr[i, max_offset : row_width + max_offset, :]
            else:
                new_mask_arr[
                    i, max_offset + o : row_width + max_offset + o, :
                ] = mask_arr[i, :, :]

    return Image.fromarray(
            np.uint8(new_mask_arr_copy if horizontal and vertical else new_mask_arr)
        ).convert("RGB")
def _apply_func_distorsion(mask, insmasks, vertical, horizontal, max_offset, func):
    """
        Apply a distorsion to an image
    """
    allm=[_apply_core(m,vertical, horizontal, max_offset, func) for m in [mask]+insmasks];
    return allm[0],allm[1:];


def sin(mask, insmasks, vertical=False, horizontal=False):
    """
        Apply a sine distorsion on one or both of the specified axis
    """

    max_offset = int(mask.height ** 0.5)

    return _apply_func_distorsion(
        mask,insmasks,
        vertical,
        horizontal,
        max_offset,
        (lambda x: int(math.sin(math.radians(x)) * max_offset)),
    )


def cos(mask, insmasks, vertical=False, horizontal=False):
    """
        Apply a cosine distorsion on one or both of the specified axis
    """

    max_offset = int(mask.height ** 0.5)

    return _apply_func_distorsion(
        mask,insmasks,
        vertical,
        horizontal,
        max_offset,
        (lambda x: int(math.cos(math.radians(x)) * max_offset)),
    )


def random(mask, insmasks, vertical=False, horizontal=False):
    """
        Apply a random distorsion on one or both of the specified axis
    """

    max_offset = int(mask.height ** 0.4)

    return _apply_func_distorsion(
        mask,insmasks,
        vertical,
        horizontal,
        max_offset,
        (lambda x: rnd.randint(0, max_offset)),
    )
