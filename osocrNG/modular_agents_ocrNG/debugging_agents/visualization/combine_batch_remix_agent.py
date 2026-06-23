import cv2
import numpy as np
import math

from neko_sdk.neko_framework_NG.UAE.neko_modwrapper_agent import neko_module_wrapping_agent
from torch.distributions import Categorical
from neko_sdk.cfgtool.argsparse import neko_get_arg
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from torch.nn import functional as trnf
def compile_list_cvims(image_list: list, canvas_width: int = 960) -> np.ndarray:
    """
    Resizes a list of OpenCV images to a uniform height and tiles them
    onto a canvas with a fixed width, placing as many images as possible in each row.

    The uniform height is determined by the minimum original height, but
    must not be less than half the maximum original height among all images.

    Args:
        image_list: A list of NumPy arrays (OpenCV images) in BGR format.
        canvas_width: The desired fixed width of the final output canvas.
                      Defaults to 960 pixels.

    Returns:
        A single NumPy array (OpenCV image) representing the tiled canvas,
        or an empty NumPy array if the input list is empty.
    """
    if not image_list:
        print("Input image list is empty.")
        return np.array([])

    # --- Step 1: Determine the Uniform Target Height (H_target) ---

    # 1a. Calculate the original height for every image
    original_heights = [img.shape[0] for img in image_list]

    # 1b. Find the minimum original height (H_min)
    h_min_orig = min(original_heights)

    # 1c. Find the minimum acceptable height (H_min_acceptable) based on the constraint:
    # H_target must be >= 1/2 of the largest original height (max(H_orig))
    h_max_orig = max(original_heights)
    h_min_acceptable = math.ceil(h_max_orig / 2)

    # 1d. The final target uniform height (H_target) is the maximum of the
    # naturally smallest height (H_min_orig) and the constrained minimum height (H_min_acceptable).
    H_target = max(h_min_orig, h_min_acceptable)

    print(f"Original heights: {original_heights}")
    print(f"Minimum acceptable height (based on max original height / 2): {h_min_acceptable}")
    print(f"Final determined uniform height (H_target): {H_target} pixels")

    # --- Step 2: Resize All Images to the Target Height ---
    resized_images = []

    for img in image_list:
        # Calculate new width to maintain aspect ratio
        orig_height, orig_width = img.shape[:2]

        # New width must be an integer
        W_new = int(round(orig_width * (H_target / orig_height)))

        # Resize using INTER_AREA for shrinking, or INTER_LINEAR for slight enlarging
        interpolation = cv2.INTER_AREA if H_target < orig_height else cv2.INTER_LINEAR

        resized_img = cv2.resize(img, (W_new, H_target), interpolation=interpolation)
        resized_images.append(resized_img)

    print(f"Total number of images resized: {len(resized_images)}")

    # --- Step 3: Tile Images into Rows ---

    tiled_rows = []
    current_row_images = []
    current_row_width = 0

    for img in resized_images:
        img_width = img.shape[1]

        # Check if adding the current image exceeds the canvas width
        if current_row_width + img_width <= canvas_width:
            # Image fits, add it to the current row
            current_row_images.append(img)
            current_row_width += img_width
        else:
            # Image does not fit, finalize the current row
            if current_row_images:
                # Horizontally stack images in the completed row
                row_img = cv2.hconcat(current_row_images)
                tiled_rows.append(row_img)

            # Start a new row with the current image
            current_row_images = [img]
            current_row_width = img_width

    # Handle the last row remaining after the loop
    if current_row_images:
        row_img = cv2.hconcat(current_row_images)
        tiled_rows.append(row_img)

    # --- Step 4: Pad Rows and Vertically Concatenate ---

    final_rows_padded = []
    # Use a black background color (0, 0, 0)
    background_color = [0, 0, 0]

    for row in tiled_rows:
        row_height, row_width = row.shape[:2]

        # Calculate padding needed on the right
        padding_needed = canvas_width - row_width

        if padding_needed > 0:
            # Create a padding matrix
            padding = np.full((row_height, padding_needed, 3), background_color, dtype=np.uint8)
            # Concatenate the row image and the padding
            padded_row = cv2.hconcat([row, padding])
        else:
            padded_row = row

        final_rows_padded.append(padded_row)

    # Vertically stack all the padded rows to form the final canvas
    final_canvas = cv2.vconcat(final_rows_padded)

    print(f"Final canvas size: {final_canvas.shape[1]}x{final_canvas.shape[0]} (W x H)")
    return final_canvas


class neko_combined_show_agent(neko_module_wrapping_agent):
    INPUT_image_list="image_list";
    OUTPUT_image_canvas="output_canvas";
    PARAM_show_each="show_each";
    PARAM_winname="winname";
    PARAM_TIME_OUT="timeout";

    def set_mod_io(this, iocvt_dict, modcvt_dict):
        this.image_list = this.register_input(this.INPUT_image_list, iocvt_dict);
        this.image_canvas = this.register_output(this.OUTPUT_image_canvas, iocvt_dict);
        pass;

    def set_etc(this, params):
        this.TIME_OUT = neko_get_arg(this.PARAM_TIME_OUT, params);
        this.show_each = neko_get_arg(this.PARAM_show_each, params);
        this.winname = neko_get_arg(this.PARAM_winname, params);
        cv2.namedWindow(this.winname,0);
        pass;

    def take_action(this, workspace: neko_workspace, environment: neko_environment):
        image_list = workspace.get(this.image_list);
        canvas=compile_list_cvims(image_list,960);
        workspace.add(this.image_canvas,canvas);
        if(workspace.get_iter()%this.show_each==0):
            cv2.imshow(this.winname,canvas);
            cv2.waitKey(this.TIME_OUT);
        return workspace, environment;


    @classmethod
    def get_agtcfg(cls,
                   image_list,
                   image_canvas,
                   TIME_OUT=1, show_each=1, winname="meow"
                   ):
        return {"agent": cls,
                "params": {"iocvt_dict": {cls.INPUT_image_list: image_list, cls.OUTPUT_image_canvas: image_canvas},
                           cls.PARAM_TIME_OUT: TIME_OUT, cls.PARAM_show_each: show_each, cls.PARAM_winname: winname,
                           "modcvt_dict": {}}}


if __name__ == '__main__':
    neko_combined_show_agent.print_default_setup_scripts()

