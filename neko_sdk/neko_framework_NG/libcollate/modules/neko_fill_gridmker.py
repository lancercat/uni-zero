import os.path

from torch import nn
from neko_sdk.cfgtool.argsparse import neko_get_arg
import torch

import torch.nn.functional as trnf

class neko_fill_fixed_size_static(nn.Module):
    PARAM_template_w="template_w";
    PARAM_template_h="template_h";

    def mkgrid(this, w, h):
        xs = torch.linspace(-1, 1, w, dtype=torch.float32)
        ys = torch.linspace(-1, 1, h, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0)
        return grid.permute(0,3,1,2);


    def __init__(this,params):
        super().__init__();
        this.template_w=neko_get_arg(this.PARAM_template_w,params);
        this.template_h=neko_get_arg(this.PARAM_template_h,params);
        this.grid=nn.Parameter(
            this.mkgrid(this.template_w,this.template_h),
            requires_grad=False
        );
    def forward(this,raw_src,thumb_map,thumb_ten):
        # just return the raw grids
        return this.grid.repeat([len(raw_src),1,1,1]);


if __name__ == '__main__':
    import glob
    import cv2
    import numpy as np

    def verify(test_image_root, template_hw=(128, 128)):
        """
        Verifies the functionality of the neko_fill_fixed_size_static module.
        It loads images from a specified root, resizes them, and displays the
        collated results using OpenCV.

        Args:
            test_image_root (str): Path to the directory containing test images,
                                   or a glob pattern (e.g., "images/*.jpg").
            template_hw (tuple): A tuple (width, height) specifying the target output dimensions.
        """
        print(f"--- Starting Verification with target size: {template_hw[0]}x{template_hw[1]} ---")
        print(f"Looking for images in: {test_image_root}")

        # 1. Find image paths using glob
        image_paths = glob.glob(os.path.join(test_image_root,"*.jpg"))
        if not image_paths:
            print(f"No images found at '{test_image_root}'. Please ensure the path is correct and contains images.")
            print("To run this example, you might need to create a folder (e.g., 'test_images')")
            print("and place some image files (e.g., .jpg, .png) inside it, then call:")
            print("verify(test_image_root='test_images/*.jpg')")
            return

        input_images_tensors = []
        original_images_for_display = []

        # 2. Load and preprocess images
        for i, img_path in enumerate(image_paths):
            print(f"Loading image {i+1}/{len(image_paths)}: {img_path}")
            img_np = cv2.imread(img_path)
            if img_np is None:
                print(f"Warning: Could not load image {img_path}. Skipping.")
                continue

            # Store original for display later
            original_images_for_display.append(img_np)

            # Convert BGR to RGB (OpenCV default is BGR)
            img_rgb_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

            # Convert NumPy array (H, W, C) to PyTorch tensor (C, H, W) and normalize to [0, 1]
            img_tensor = torch.from_numpy(img_rgb_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # Add batch dimension (1, C, H, W)
            input_images_tensors.append(img_tensor)

        if not input_images_tensors:
            print("No valid images were loaded for processing.")
            return

        # 3. Prepare parameters for the module.
        params = {
            neko_fill_fixed_size_static.PARAM_template_w: template_hw[0],
            neko_fill_fixed_size_static.PARAM_template_h: template_hw[1]
        }

        # 4. Instantiate the neko_fill_fixed_size_static module.
        try:
            resizer = neko_fill_fixed_size_static(params)
            print(f"Module initialized with template_w={resizer.template_w}, template_h={resizer.template_h}")

            # 5. Pass the list of input image tensors through the forward method.
            output_images_tensor = resizer.forward(input_images_tensors)
            print(f"Output images tensor shape: {output_images_tensor.shape}")

            # 6. Convert output tensor back to NumPy for display
            # Permute from (B, C, H, W) to (B, H, W, C) and scale to [0, 255]
            output_images_np = (output_images_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

            # 7. Display original and resized images
            for i in range(output_images_np.shape[0]):
                original_img = original_images_for_display[i]
                resized_img = output_images_np[i]

                # Convert RGB back to BGR for OpenCV display
                resized_img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)

                # Create a canvas to show original and resized side-by-side
                # Ensure they have the same number of channels (3 for RGB/BGR)
                if len(original_img.shape) == 2: # Grayscale
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
                elif original_img.shape[2] == 4: # RGBA
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGRA2BGR)
                original_img=cv2.resize(original_img,(128,128))
                # Resize original to match resized height for display if needed, or pad
                # For simplicity, let's just show them separately or pad original if smaller
                # For a cleaner display, we'll collate them horizontally.
                # Pad original image to match height of resized image for horizontal stacking
                h_orig, w_orig = original_img.shape[:2]
                h_resized, w_resized = resized_img_bgr.shape[:2]

                if h_orig < h_resized:
                    padding_needed = h_resized - h_orig
                    original_img_padded = cv2.copyMakeBorder(
                        original_img,
                        0, padding_needed, 0, 0, # top, bottom, left, right
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0] # Black padding
                    )
                else:
                    original_img_padded = original_img

                # Resize original image to a reasonable display size if it's too large
                max_display_width = 400
                if original_img_padded.shape[1] > max_display_width:
                    scale = max_display_width / original_img_padded.shape[1]
                    original_img_padded = cv2.resize(original_img_padded, (max_display_width, int(original_img_padded.shape[0] * scale)))

                # Ensure resized_img_bgr is also padded/resized to match heights for concatenation
                if original_img_padded.shape[0] != resized_img_bgr.shape[0]:
                    resized_img_bgr = cv2.resize(resized_img_bgr, (resized_img_bgr.shape[1], original_img_padded.shape[0]))


                collated_image = np.hstack((original_img_padded, resized_img_bgr))
                cv2.imshow(f"Original vs Resized Image {i+1}", collated_image)
                cv2.imshow(f"Resized Image {i+1}", collated_image)

            cv2.waitKey(0) # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows() # Close all OpenCV windows

            print("Verification successful: Images processed and displayed.")

        except Exception as e:
            print(f"An error occurred during verification: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

        print("--- Verification Finished ---")
    verify("/home/lasercat/writebuffer/handy_tests/");