import os
import shutil

MISC={
        "misc":{
            "misc" : ["b1l1NF-1024-hindi-cntralgn-lcam-ctc-32-128",
"b1l1NF-1024-kannada-cntralgn-lcam-ctc-32-128",
"b1l1NF-1024-malayalam-cntralgn-lcam-ctc-32-128",
"b1l1NF-1024-marathi-cntralgn-lcam-ctc-32-128",
"b1l1NF-1024-odia-cntralgn-lcam-ctc-32-128",
"b1l1NF-1024-punjabi-cntralgn-lcam-ctc-32-128",
"b1l1NF-1024-tamil-cntralgn-lcam-ctc-32-128",
"b1l1NF-1024-telugu-cntralgn-lcam-ctc-32-128",
"b1m1nf-1024-hustobc-lsctfrc-abiaug-48-48",
"b1m3nf-1024-hustobcZ-lsctfrc-abiaug-48-48-re"],

        }

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
    all_dicts = [MISC]

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
