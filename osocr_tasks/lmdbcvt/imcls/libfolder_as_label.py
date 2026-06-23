import glob
import os.path
import shutil
import tqdm
from PIL import Image


def cache_generic_folder(root_dir,target_root, split, extensions):
    """
    extensions: list of strings, e.g., ["*.jpg", "*.png", "*.jpeg"]
    """
    split_path = os.path.join(root_dir, split)
    class_folders = glob.glob(os.path.join(split_path, "*"))
    target_split = os.path.join(target_root, split)

    shutil.rmtree(target_split, ignore_errors=True)
    os.makedirs(target_split, exist_ok=True)

    for cf in tqdm.tqdm(class_folders, desc="Caching Data"):
        class_id = os.path.basename(cf)
        img_dir = cf

        # Gather all images matching any pattern in the list
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(img_dir, ext)))

        # Create target: target/train/[class_id]/0/
        save_dir = os.path.join(target_split, class_id, "0")
        os.makedirs(save_dir, exist_ok=True)

        for idx, img_path in enumerate(images):
            # Standardizing name to your expected format
            save_path = os.path.join(save_dir, f"img_doc_0_{idx}.png")
            try:
                with Image.open(img_path) as img:
                    img.convert("RGB").save(save_path)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}");



def cache_folder_split(root_dir, target_root, split):
    """
    Handles Tiny ImageNet structure:
    - train: root/train/[class]/images/*.JPEG
    - val: root/val/images/*.JPEG + val_annotations.txt
    """


    if split == "train":
        cache_generic_folder(root_dir,target_root,split,["*.JPEG"])
        # Structure: train/n01443537/images/n01443537_0.JPEG

    elif split == "val":
        split_path = os.path.join(root_dir, split)
        target_split = os.path.join(target_root, split)

        shutil.rmtree(target_split, ignore_errors=True)
        os.makedirs(target_split, exist_ok=True)
        # Structure: val/images/*.JPEG and val_annotations.txt
        anno_file = os.path.join(split_path, "val_annotations.txt")
        img_dir = os.path.join(split_path, "images")

        with open(anno_file, 'r') as f:
            lines = f.readlines()

        lcnt = {}
        for line in tqdm.tqdm(lines, desc="Caching Val"):
            parts = line.strip().split('\t')
            img_name = parts[0]
            class_id = parts[1]

            if class_id not in lcnt:
                lcnt[class_id] = 0
                os.makedirs(os.path.join(target_split, class_id, "0"), exist_ok=True)

            src_path = os.path.join(img_dir, img_name)
            save_path = os.path.join(target_split, class_id, "0", f"img_doc_0_{lcnt[class_id]}.png")

            with Image.open(src_path) as img:
                img.convert("RGB").save(save_path)
            lcnt[class_id] += 1
