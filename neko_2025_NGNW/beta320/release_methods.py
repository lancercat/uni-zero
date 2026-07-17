import os
import shutil

STRA={
        "Indc":{
            "CamCtc" : ["b1h1NF-1024-indc2-cntralgn-lcam-ctc-32-128","b1h1NF-1024-indc2-cntralgn-lcam-ctc-32-128-run2"],
            "CamLpos": ["b1h1NF-1024-indc2-cntralgn-lcam-lposReal-32-128","b1h1NF-1024-indc2-cntralgn-lcam-lposReal-32-128-run2"],
            "SqzCtc": ["b1h1NF-1024-indc2-cntralgn-sqz-lctc-32-128","b1h1NF-1024-indc2-cntralgn-sqz-lctc-32-128-run2"]
        },
        "Ostr": {
            "CamCtc" :["b1NF-1024-ostr-cntralgn-lcam-lpos-32-128","b1NF-1024-ostr-cntralgn-lcam-lpos-32-128-run2"],
            "CamLpos": ["b2NF-1024-ostr-cntralgn-lcam-lposReal-32-128",
                        "b2NF-1024-ostr-cntralgn-lcam-lposReal-32-128-run2"],
            "SqzCtc": ["b1dNF-1024-ostr-cntralgn-sqz-lctc-32-128","b1dNF-1024-ostr-cntralgn-sqz-lctc-32-128-run2"]
        }
    }
CFLA={
    "Indc": {
        "BigBuc": ["b1h4NF-1024-indc-cntralgn-lcam-ctc-32-128"],
        "ConflCamCtc": ["b1h1NF-1024-indc2-cntralgn-lcam-ctc-32-128"],
    }
}
LSCTA={
     "Ostr": {
    "Xbatch": ["b1NFXB4-1024-ostr-cntralgn-lcam-lpos-32-128", "b1NFXB4-1024-ostr-cntralgn-lcam-lpos-32-128-run2"],
    "ShufSA": ["b1NF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT","b1NF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT-run2"],
        "FRC": ["b1iNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctFRC_RealNI64",
               "b1iNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctFRC_RealNI64-run2"],
    "ShufFRC": ["b1jNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT-lsctFRC_RealNI64","b1jNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT-lsctFRC_RealNI64-run2"],
},
    "indc": {
        "ShufSA": ["b1h2NF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT",
                   "b1h2NF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT-run2"],

        "FRC": ["b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctFRC_RealNI64",
                      "b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctFRC_RealNI64-run2"],
        "ShufFRC": ["b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT-lsctFRC_RealNI64",
                      "b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT-lsctFRC_RealNI64-run2"],
    }
}
UNI = {
    "Uni": {"Uni": ["b1h5NF-1024-all-cntralgn-lcam-ctc-32-128-SASTT",
                    "b1h5NF-1024-all-cntralgn-lcam-ctc-32-128-SASTT-run2"]},
}

def cp(methods_list, src_root, dst_root):
    """
    Copies directory folders from src_root to dst_root based on a list of folder names.
    Creates dst_root if it doesn't already exist.
    """
    # Ensure destination root exists
    os.makedirs(dst_root, exist_ok=True)

    for folder in methods_list:
        src_path = os.path.join(src_root, folder)
        dst_path = os.path.join(dst_root, folder)

        # Check if the source folder actually exists before trying to copy
        if os.path.exists(src_path) and os.path.isdir(src_path):
            try:
                # shutil.copytree copies entire directories recursively
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Successfully copied: {folder}")
            except Exception as e:
                print(f"Error copying {folder}: {e}")
        else:
            print(f"Warning: Source folder does not exist or is not a directory: {src_path}")


if __name__ == '__main__':
    # Combine all target dictionaries into a list
    all_dicts = [STRA,CFLA,LSCTA]

    # Use a set to collect only unique method names
    unique_methods = set()

    for current_dict in all_dicts:
        for strategy, methods_dict in current_dict.items():
            # methods_dict.keys() gives us keys like "CamCtc", "CamLpos", etc.
            for f in methods_dict:
                for r in methods_dict[f]:
                    unique_methods.add(r)

    # Convert to a sorted list for cleaner display
    methods_list = sorted(list(unique_methods))

    cp(methods_list,"/run/media/lasercat/writebuffer/hydra_saves/","/run/media/lasercat/320-eccv/hydra_saves");
