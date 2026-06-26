import os
from jiwer import cer, wer


def fn2pk(filename):
    """
    Extracts the unique ID/Primary Key from the filename.
    Example: 'sample_001-gt.txt' -> 'sample_001'
    """
    return filename.split(".")[0].split("-")[0];

def calculate_metrics(gt_folder, pred_folder):
    # Get lists of files and filter for .txt
    gt_files = {fn2pk(f): f for f in os.listdir(gt_folder) if f.endswith('.txt')}
    pred_files = {fn2pk(f): f for f in os.listdir(pred_folder) if f.endswith('.txt')}

    # Find common IDs present in both folders
    # common_ids = sorted(list(set(gt_files.keys()) & set(pred_files.keys())))
    common_ids = sorted(list(set(gt_files.keys())))

    print(f"{'Sample ID':<25} | {'CER':<10} | {'WRR':<10}")
    print("-" * 50)

    total_cer = 0
    total_wer = 0
    count = 0

    for pk in common_ids:
        # Load GT text
        with open(os.path.join(gt_folder, gt_files[pk]), 'r', encoding='utf-8') as f:
            gt_text = f.read().strip()
        try:
            # Load Pred text
            with open(os.path.join(pred_folder, pred_files[pk]), 'r', encoding='utf-8') as f:
                pred_text = f.read().strip()#.replace("⑨","")
        except:
            pred_text="";
        # Calculate metrics
        error_char = cer(gt_text, pred_text)
        error_word = min(wer(gt_text, pred_text),1)
        wrr = max(0, 1 - error_word)

        print(f"{pk:<25} | {error_char:.4f}     | {wrr:.4f}")

        total_cer += error_char
        total_wer += error_word
        count += 1

    if count > 0:
        avg_cer = total_cer / count
        avg_wrr = 1 - (total_wer / count)
        print("-" * 50)
        print(f"{'AVERAGE':<25} | {avg_cer:.4f}     | {avg_wrr:.4f}",count)
    else:
        print("No matching file IDs found between the two folders.")

if __name__ == '__main__':
    gt_path="/run/media/lasercat/writebuffer/tmp/hydra_results_uni_gt/yi50_gt/";

    # Paths
    # gt_path = '/run/media/lasercat/writebuffer/tmp/synthyi_5k/ours-vs-gpt2/gt/'
    # pred_path = '/run/media/lasercat/writebuffer/tmp/synthyi_5k/ours-vs-gpt2/ours-uni/'

    # gt_path="/run/media/lasercat/writebuffer/tmp/hydra_results_uni_gt/syn_yi_hori_gzsl/";
    # pred_path="/run/media/lasercat/writebuffer/tmp/hydra_results_uni_pred/syn_yi_hori_gzsl/";
    #

    pred_path="/run/media/lasercat/writebuffer/tmp/synthyi_5k/predicts/qwen3.6_27b/";
    pred_path="/home/lasercat/kn/data/eccv_rebuttal/predicts_nocset/gemini-3.1-flash-lite-preview/"
    calculate_metrics(gt_path, pred_path)