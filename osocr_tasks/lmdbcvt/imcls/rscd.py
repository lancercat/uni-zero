
from export_cache import export_split


from libfolder_as_label import cache_generic_folder
if __name__ == '__main__':
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt

    sroot="/run/media/lasercat/320-eccv/results/wiertenc/extract/rscd/"
    droot="/run/media/lasercat/320-eccv/results/rscd/"

    # dataset = datasets.load_dataset("bentrevett/caltech-ucsd-birds-200-2011")
    split="train";
    cache_generic_folder(sroot,droot, split,["*.jpg", "*.png", "*.jpeg"]);

    split="vali_20k";
    cache_generic_folder(sroot,droot, split,["*.jpg", "*.png", "*.jpeg"]);

    split="test_50k";
    cache_generic_folder(sroot,droot, split,["*.jpg", "*.png", "*.jpeg"]);


    tarroot=""

    # export_split(droot, "train","tin-200");
    # export_split(droot, "val","tin-200");
