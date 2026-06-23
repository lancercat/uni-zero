import torch

from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from osocrNG.data_utils.neko_im_label_lmdb_holder import neko_im_label_lmdb_holder
import tqdm
def get_all_text(dbroot):
    holder=neko_im_label_lmdb_holder({neko_im_label_lmdb_holder.PARAM_root:dbroot});
    all_labels=[];
    for id in tqdm.tqdm(holder.all_valid_indexes()):
        item=holder.fetch_item(id);
        if(item is not None):
            all_labels.append(item[RN.LABEL]);
    return all_labels;
if __name__ == '__main__':
    jptxt=get_all_text("/home/lasercat/ssddata/mlttrjp_hv");
    krtxt=get_all_text("/home/lasercat/ssddata/mlttrkr_hori/");
    lsvt_txt=get_all_text("/home/lasercat/ssddata/lsvtdb_seen_NG");
    torch.save({
        "jpn":jptxt,
        "kr":krtxt,
        "seen":lsvt_txt},"/run/media/lasercat/ssddata2/synth/corpus.pt");
    pass;
