import json;
import os;
import shutil;

import cv2;
import numpy as np;
import torch
import yaml  # Import for finalize_path and mkyaml
import random  # Import for finalize_path

from neko_sdk.lmdb_wrappers.im_lmdb_wrapper import im_lmdb_wrapper


def final_fix(cou, angguess):
    c, (w, h), t = cou;
    if angguess > 60:
        return c, (h, w), t - 90;
    if angguess < -60:
        return c, (h, w), t + 90;
    return cou;


def fix_rrect(cou, angguess):
    c, (w, h), t = cou;
    if (np.abs(angguess - t) > 120):
        return None;
    if (np.abs(angguess - t) > 60):
        if (angguess > t):
            t += 90;
            return final_fix((c, (h, w), t), angguess);
        else:
            t -= 90;
            return final_fix((c, (h, w), t), angguess);
    return final_fix(cou, angguess);


def handle_char_gt(cdict):
    rect = cdict["adjusted_bbox"];
    text = cdict["text"];
    attr = cdict["attributes"]
    polygon=cdict["polygon"];
    # NOTE: If a polygon was available, e.g., cdict["adjusted_polygon"],
    # we would also return it here to generate the binary mask.
    return text, rect, attr,polygon;


def handle_word_gt(wdict):
    texts = [];
    boxes = [];
    attrs = [];
    polygons=[];
    for ch in wdict:
        t, b, a,p = handle_char_gt(ch);
        texts.append(t);
        boxes.append(b);
        attrs.append(a);
        polygons.append(p);
    return texts, boxes, attrs,polygons;


def handle_file_gt(fdict):
    texts, boxes, attrs,polygons = [], [], [],[];
    for item in fdict["annotations"]:
        text, box, attr,polys = handle_word_gt(item);
        texts += text;
        boxes += box;
        attrs += attr;
        polygons+=polys;
    return texts, boxes, attrs,polygons;


def subimage(image, box):
    """Crops an ROI from the image given a bounding box [l, t, w, h]."""
    l, t, w, h = box;
    l = int(l);
    r = int(l + w);
    t = int(t);
    b = int(t + h);
    # Ensure coordinates are within image bounds
    imh, imw = image.shape[:2]
    l = max(0, l)
    r = min(imw, r)
    t = max(0, t)
    b = min(imh, b)
    if l >= r or t >= b:
        return None  # Invalid crop
    return image[t:b, l:r, :];


def yolo_line(image, box, text, cdic):
    l, t, w, h = box
    imh, imw = image.shape[:2]
    if text not in cdic:
        cdic[text] = len(cdic)
    return [
        str(cdic[text]),
        str((l + w / 2) / imw),
        str((t + h / 2) / imh),
        str(w / imw),
        str(h / imh),
    ], cdic


def handle_file(fdict, dataroot, dst, cdic, roi_dst):
    """
    Processes a single file, converting its annotations to YOLO format
    and saving ROIs in class-specific subfolders.

    Args:
        fdict (dict): A dictionary containing file information, including the file name.
        dataroot (str): The root directory where the image files are located.
        dst (str): The destination directory to save the YOLO annotation files.
        cdic (dict): A dictionary mapping class labels to integer IDs.
        roi_dst (str): The root directory to save character ROIs.

    Returns:
        dict: The updated class dictionary (cdic).
    """
    file_name = os.path.join(dataroot, fdict["file_name"])
    base_name = os.path.splitext(os.path.basename(fdict["file_name"]))[0]
    im = cv2.imread(file_name)
    if im is None:
        print(f"Warning: Could not read image file: {file_name}")
        return cdic

    try:
        texts, boxes, attrs,polygons = handle_file_gt(fdict)
    except NameError:
        print("Error: handle_file_gt is not defined. Please provide the implementation.")
        return cdic

    targets = []
    for b in range(len(boxes)):
        if boxes[b] is None:
            continue
        box = boxes[b]
        text = texts[b]
        poly_coords=polygons[b];
        attr=attrs[b];
        # --- ROI Saving Logic ---
        roi_img = subimage(im, box)
        target, cdic = yolo_line(im, box, text, cdic)
        targets.append(target)
        if roi_img is not None:
            # Create a class-specific subfolder in roi_dst
            class_folder = os.path.join(roi_dst, str(cdic[text]));
            os.makedirs(class_folder, exist_ok=True)

            # Save the ROI image
            # Naming convention: {original_base_name}_{box_index}.jpg
            roi_path = os.path.join(class_folder, f"{base_name}_{b}.jpg")
            attr_path = os.path.join(class_folder, f"{base_name}_{b}_attr.txt")
            with open(attr_path,"w+") as fp:
                fp.writelines([_+"\n" for _ in attr]);
            cv2.imwrite(roi_path, roi_img)

            # NOTE: To save a binary mask, the polygon coordinates would be needed.
            # If the GT had a polygon (e.g., poly = attrs[b].get("polygon")),
            # you would draw it onto a black image of size (h, w) of the bounding box.
            mask = np.zeros(roi_img.shape[:2], dtype=np.uint8)
            l,t,w,h=box;
            local_poly_coords = np.array(poly_coords) - [l, t]

            # 2. **CLAMP** coordinates to the mask's bounds [0, w-1] x [0, h-1]
            local_poly_coords[:, 0] = np.clip(local_poly_coords[:, 0], 0, w - 1)
            local_poly_coords[:, 1] = np.clip(local_poly_coords[:, 1], 0, h - 1)

            # 3. Ensure coordinates are integers for fillPoly
            local_poly_coords = local_poly_coords.astype(np.int32)

            cv2.fillPoly(mask, [local_poly_coords], 255);
            cv2.imwrite(os.path.join(class_folder, f"{base_name}_{b}_mask.png"), mask)

        # --- YOLO Annotation Logic ---

    # Construct the output file path.
    # Note: fdict["file_name"] might contain a path, so we use os.path.basename
    gt_base_name = os.path.splitext(os.path.basename(fdict["file_name"]))[0]
    output_path = os.path.join(dst, f"{gt_base_name}.txt")

    # Write the YOLO targets to the file.
    with open(output_path, "w") as f:
        for target in targets:
            f.write(" ".join(target) + "\n")

    return cdic


def make_ctwch(trpath, dataroot, dst, roi_dst):
    """
    Reads ground truth, processes files, creates YOLO annotations, and saves ROIs.

    Args:
        trpath (str): Path to the JSONL ground truth file.
        dataroot (str): Root directory for image files.
        dst (str): Destination directory for YOLO label .txt files.
        roi_dst (str): Destination root directory for ROI images.
    """
    jdicts = []
    with open(trpath, "r") as fp:
        for l in fp:
            jdict = json.loads(l);
            jdicts.append(jdict);

    # Ensure the ROI destination root exists
    os.makedirs(roi_dst, exist_ok=True)

    cdic = {}
    for fid in range(len(jdicts)):
        if (fid % 0x71 == 0):
            print(fid, " of ", len(jdicts));
        # Pass the roi_dst argument here
        cdic = handle_file(jdicts[fid], dataroot, dst, cdic, roi_dst);

    # Save the character dictionary for later use in mkyaml
    # Assuming 'dst' is the 'labels' folder and its parent is the intermediate root
    cdic_save_path = os.path.join(os.path.dirname(dst), "dict.pt")
    rcdict={cdic[k]:k  for k in cdic };
    with open(os.path.join(dst,"../rcdict.json"),"w+") as fp:
        json.dump(rcdict,fp);
    with open(os.path.join(dst,"../cdict.json"),"w+") as fp:
        json.dump(cdic,fp);
    torch.save(cdic, cdic_save_path)
    print(f"Character dictionary saved to: {cdic_save_path}")
    pass;


# --- Unmodified functions for completeness, kept here to prevent NameErrors ---
def mkyaml(src, dst):
    # Assuming the structure is: src/dict.pt, src/images, src/labels
    cdic = torch.load(os.path.join(src, "dict.pt"))

    # Invert the dictionary to get class index to name mapping for YOLO's YAML
    names = {v: k for k, v in cdic.items()}

    # --- Step 5: Create the dataset.yaml file ---
    dataset = {
        "path": dst,
        "train": "images/train",  # Path to training images relative to 'path'
        "val": "images/val",  # Path to validation images relative to 'path'
        "test": "images/test",  # Path to test images relative to 'path'
        "nc": len(names),  # Number of classes
        "names": names  # Dictionary of class names
    }

    yaml_path = os.path.join(dst, "dataset.yaml")
    print(f"Creating dataset.yaml at {yaml_path}...")
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset, f, sort_keys=False)

    print("Dataset preparation complete.")


def make_split(src, split_ratio=[0.7, 0.2, 0.1]):
    """
    Splits the list of valid image/label pairs into train, val, and test prefixes.

    Args:
        src (str): The root directory containing 'images' and 'labels' subfolders.
        split_ratio (list): A list of three floats [train, val, test] summing to 1.0.

    Returns:
        tuple: (train_files, val_files, test_files) - lists of image base names.
    """
    if sum(split_ratio) != 1.0:
        # Use close enough for float comparison
        if abs(sum(split_ratio) - 1.0) > 1e-6:
            raise ValueError("The sum of the split ratios must equal 1.0.")
    if len(split_ratio) != 3:
        raise ValueError("split_ratio must be a list of three floats [train, val, test].")

    # --- Step 1: List all valid images ---
    image_dir = os.path.join(src, "images")
    labels_dir = os.path.join(src, "labels")

    if not os.path.exists(image_dir) or not os.path.exists(labels_dir):
        print("Error: 'images' or 'labels' directory not found in src.")
        return [], [], []

    # Get a list of all image files (without extensions)
    # Assumes images are .jpg or similar common format; checking for any file
    all_files = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # Ensure a corresponding label file exists for each image
    valid_files = []
    for file_name in all_files:
        # Assuming label files are .txt
        if os.path.exists(os.path.join(labels_dir, file_name + ".txt")):
            valid_files.append(file_name)

    if not valid_files:
        print("No valid image/label pairs found in the source directory.")
        return [], [], []

    # Shuffle the list for a random split
    random.seed(42) # Set a seed for reproducibility
    random.shuffle(valid_files)

    # --- Step 2: Split the data based on the ratio ---
    total_count = len(valid_files)
    train_count = int(total_count * split_ratio[0])
    val_count = int(total_count * split_ratio[1])
    test_count = total_count - train_count - val_count  # Remaining files go to test

    train_files = valid_files[:train_count]
    val_files = valid_files[train_count:train_count + val_count]
    test_files = valid_files[train_count + val_count:]

    print(f"Split calculated: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    return train_files, val_files, test_files


def split_yolo(src, dst, train_imprfx_list, val_imprfx_list, test_imprfx_list):
    """
    Copies image and label files to the YOLO-standard destination structure.

    Args:
        src (str): The intermediate root directory.
        dst (str): The final destination YOLO root directory.
        train_imprfx_list (list): List of base names (prefixes) for the training set.
        val_imprfx_list (list): List of base names (prefixes) for the validation set.
        test_imprfx_list (list): List of base names (prefixes) for the test set.
    """
    print(f"Copying files from {src} to final YOLO structure in {dst}...")

    # --- Step 1: Prepare directories ---
    image_dir = os.path.join(src, "images")
    labels_dir = os.path.join(src, "labels")

    # Remove existing destination directory to start fresh
    if os.path.exists(dst):
        shutil.rmtree(dst)

    # Create destination structure
    os.makedirs(os.path.join(dst, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dst, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dst, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(dst, "labels", "test"), exist_ok=True)

    splits = {
        "train": train_imprfx_list,
        "val": val_imprfx_list,
        "test": test_imprfx_list
    }

    # --- Step 2: Copy the files ---
    for split_name, file_list in splits.items():
        print(f"Copying {len(file_list)} files for {split_name} split...")
        for file_name in file_list:
            # Copy image (assuming .jpg)
            src_image_path = os.path.join(image_dir, file_name + ".jpg")
            dst_image_path = os.path.join(dst, "images", split_name, file_name + ".jpg")
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dst_image_path)

            # Copy label (assuming .txt)
            src_label_path = os.path.join(labels_dir, file_name + ".txt")
            dst_label_path = os.path.join(dst, "labels", split_name, file_name + ".txt")
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)

    # --- Step 3: Create the dataset.yaml file ---
    mkyaml(src, dst)
    print("Dataset preparation complete.")


def get_train_val_test_roi_paths(roi_folder, train_imprfx_list, val_imprfx_list, test_imprfx_list):
    """
    Categorizes the full paths of Character ROI files (images, masks, and attributes)
    into a structured dictionary: result_paths[split_name][label_id][file_type].

    Args:
        roi_folder (str): The root directory where class-specific ROIs are saved (e.g., 'interm/rois_by_class').
        train_imprfx_list (list): List of base names (prefixes) of images belonging to the training set.
        val_imprfx_list (list): List of base names (prefixes) of images belonging to the validation set.
        test_imprfx_list (list): List of base names (prefixes) of images belonging to the test set.

    Returns:
        dict: A dictionary structured as:
              {split_name: {label_id: {file_type: [list_of_paths]}}}
    """
    import os
    from collections import defaultdict

    print(f"Starting ROI path categorization from {roi_folder}...")

    # Initialize the result dictionary using defaultdict for easy, nested creation
    # The innermost structure (list) holds the paths.
    result_paths = {
        "train": defaultdict(lambda: defaultdict(list)),
        "val": defaultdict(lambda: defaultdict(list)),
        "test": defaultdict(lambda: defaultdict(list))
    }

    # 1. Consolidate prefix lists into a mapping for fast lookup
    imprfx_map = {}
    for prefix in train_imprfx_list:
        imprfx_map[prefix] = "train"
    for prefix in val_imprfx_list:
        imprfx_map[prefix] = "val"
    for prefix in test_imprfx_list:
        imprfx_map[prefix] = "test"

    # 2. Iterate through all class folders (which are the label_ids)
    for class_id in os.listdir(roi_folder):
        label_id = class_id  # The directory name is the label_id
        class_src_dir = os.path.join(roi_folder, label_id)
        if not os.path.isdir(class_src_dir):
            continue

        # 3. Iterate through all ROI files
        for filename in os.listdir(class_src_dir):
            src_path = os.path.join(class_src_dir, filename)

            # Determine the original image base name (prefix) and file type
            base_name = ""
            file_type = None

            if filename.endswith("_mask.png"):
                # Mask file: {base_name}_{box_index}_mask.png
                try:
                    # rsplit('_', 3) to split off: [box_index], ['mask'], ['png']
                    base_name = filename.rsplit('_', 3)[0]
                    file_type = "mask"
                except IndexError:
                    continue
            elif filename.endswith("_attr.txt"):
                # Attributes file: {base_name}_{box_index}_attr.txt
                try:
                    # rsplit('_', 3) to split off: [box_index], ['attr'], ['txt']
                    base_name = filename.rsplit('_', 3)[0]
                    file_type = "attributes"
                except IndexError:
                    continue
            elif filename.endswith(".jpg"):  # Assuming image files are .jpg
                # Image file: {base_name}_{box_index}.jpg
                try:
                    # rsplit('_', 2) to split off: [box_index], ['jpg']
                    base_name = filename.rsplit('_', 2)[0]
                    file_type = "image"
                except IndexError:
                    continue
            else:
                continue  # Skip files that don't match expected patterns

            # 4. Determine the split
            split_name = imprfx_map.get(base_name)

            if split_name and file_type:
                # 5. Add the source path to the corresponding split, label_id, and file_type list
                result_paths[split_name][label_id][file_type].append(src_path)

    print("ROI path categorization complete.")

    # Convert defaultdicts back to regular dicts for clean output, if necessary
    # If using Python 3.9+, standard dicts can handle this; otherwise, this conversion is safer.
    final_result = {}
    for split, label_dict in result_paths.items():
        final_result[split] = {k: dict(v) for k, v in label_dict.items()}

    return final_result if 'final_result' in locals() else result_paths



def mkchar_lmdb(roi_path_dict, dst,rcdict_path):
    with open(rcdict_path,"r") as fp:
        rcdict=json.load(fp);
    for split in roi_path_dict:
        index={};
        lmdbpath=os.path.join(dst,split);
        w=im_lmdb_wrapper(lmdbpath);
        sdict=roi_path_dict[split];
        tot=0;
        for lid in sdict:
            text =rcdict[lid];
            index[text]=[];
            masks,attrs,imgs=sdict[lid]["mask"],sdict[lid]["attributes"],sdict[lid]["image"],
            for m,a,i in zip(masks,attrs,imgs):
                attrs=[];
                with open(a,"r") as fp:
                    attrs=[i.strip() for i in fp];
                iks, rks, tks=w.adddata_kv({"image":cv2.imread(i),"polymsk":cv2.imread(m)},{"label":text,"lang":"Chinese","attr":str(attrs)},{});
                index[text].append((iks,rks,tks));
                tot+=1;
        print(split,"total_imgs",tot);
        torch.save(index,os.path.join(lmdbpath,"index.pt"));


if __name__ == '__main__':
    # Define file paths
    j = "/run/media/lasercat/writebuffer/snowy/ctw/gtar/train.jsonl";  # Ground truth path
    dr = "/run/media/lasercat/writebuffer/snowy/ctw/itar/";  # Image root path
    interm = "/run/media/lasercat/writebuffer/snowy/ctw_yolox/";  # Intermediate YOLO root
    # Destination for YOLO label files (inside the intermediate root)
    label_dst = os.path.join(interm, "labels")
    # New destination for character ROIs, a sibling to 'images' and 'labels'
    roi_dst = os.path.join(interm, "rois_by_class")
    os.makedirs(roi_dst,exist_ok=True)
    # dst="/run/media/lasercat/writebuffer/snowy/ctw_yolo_fin/";
    # dst_win="/run/media/lasercat/writebuffer/snowy/ctw_yolo_win/";

    # Tweak: Call make_ctwch with the new roi_dst
    # Ensure all images are copied to interm/images before finalize_path
    # (This step is usually separate, assuming the images are already in dr)
    # The original script does not copy images; assuming it's done or dr is interm/images

    # The original script's make_ctwch writes labels to 'dst', which is 'interm' in the commented code.
    # To follow a standard YOLO structure where labels are in a 'labels' folder,
    # I set label_dst=os.path.join(interm, "labels").

    # For make_ctwch to work, the directory structure for images should also be set up,
    # e.g., by copying images from 'dr' to 'interm/images'. This is outside the scope
    # of the direct tweak but necessary for 'finalize_path'.

    print("--- Running make_ctwch with ROI saving ---")
    # make_ctwch(j, dr, label_dst, roi_dst);
    #
    # random.seed(0xCA71); # uss quincy II
    # tr,va,te=make_split("/run/media/lasercat/writebuffer/snowy/ctw_yolox/");
    # torch.save((tr,va,te),"/run/media/lasercat/writebuffer/snowy/ctw_yolox/split.pt");
    # (tr,va,te)=torch.load("/run/media/lasercat/writebuffer/snowy/ctw_yolox/split.pt");
    # pdict=get_train_val_test_roi_paths(roi_dst,tr,va,te);
    # torch.save(pdict,"/run/media/lasercat/writebuffer/snowy/ctw_yolox/roi_split.pt");
    roip=torch.load("/run/media/lasercat/writebuffer/snowy/ctw_yolox/roi_split.pt")
    mkchar_lmdb(roip,"/run/media/lasercat/writebuffer/snowy/ctw_yolox/lmdbs","/run/media/lasercat/writebuffer/snowy/ctw_yolox/rcdict.json");

    pass;

    # Example of how to structure the image directory manually if needed:
    # # Make an image copy folder if not exists
    # os.makedirs(os.path.join(interm, "images"), exist_ok=True)
    # # You would need a loop here to copy all images from 'dr' to 'interm/images'
    # # ...

    # # If finalize_path is used, it will restructure the dataset.
    # print("--- Running finalize_path (split and directory restructure) ---")
    # # finalize_path(interm, dst);

    # # The chunk_yolo part is commented out and requires the neko_libdet package.
    # # for split in ["train","val","test"]:
    # #     chunk_yolo(os.path.join(dst,"images",split),os.path.join(dst,"labels",split),
    # #                os.path.join(dst_win, "images", split), os.path.join(dst_win, "labels", split)
    # #                );