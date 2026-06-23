import glob
import os.path
import shutil
from export_cache import export_split
import tqdm



def cache_tiny_imagenet_split(root_dir, target_root, split):
    """
    Handles Tiny ImageNet structure:
    - train: root/train/[class]/images/*.JPEG
    - val: root/val/images/*.JPEG + val_annotations.txt
    """
    split_path = os.path.join(root_dir, split)
    target_split = os.path.join(target_root, split)

    shutil.rmtree(target_split, ignore_errors=True)
    os.makedirs(target_split, exist_ok=True)

    if split == "train":
        # Structure: train/n01443537/images/n01443537_0.JPEG
        class_folders = glob.glob(os.path.join(split_path, "*"))
        for cf in tqdm.tqdm(class_folders, desc="Caching Train"):
            class_id = os.path.basename(cf)
            img_dir = os.path.join(cf, "images")
            images = glob.glob(os.path.join(img_dir, "*.JPEG"))

            # Create target: target/train/[class_id]/0/
            save_dir = os.path.join(target_split, class_id, "0")
            os.makedirs(save_dir, exist_ok=True)

            for idx, img_path in enumerate(images):
                # Standardizing name to your expected format: img_IDFR_doc_IDFR_seq.png
                save_path = os.path.join(save_dir, f"img_doc_0_{idx}.png")
                with Image.open(img_path) as img:
                    img.convert("RGB").save(save_path)

    elif split == "val":
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

if __name__ == '__main__':
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    sroot="/run/media/lasercat/writebuffer/tmp/tin200/tiny-imagenet-200/"
    droot="/run/media/lasercat/writebuffer/tmp/tin-200/"

    # # dataset = datasets.load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
    # split="train";
    # tarroot=""
    # cache_tiny_imagenet_split(sroot,droot, split);
    #
    # split="val";
    # cache_tiny_imagenet_split(sroot,droot, split);

    trd=export_split(droot, "train","tin-200");
    export_split(droot, "val","tin-200",trd);
