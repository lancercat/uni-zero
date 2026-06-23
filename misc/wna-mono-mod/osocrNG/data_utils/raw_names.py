# not all stuff appear in your model,
# but if it appears, it goes with the listed name.
class raw_data_item_names:
    IMAGE="image";
    PREAUG="preaug";
    LABEL="label";
    MASKS="masks"; # if semantic segmentation mask applicable
    BEACON="beacon";
    SIZE="size";
    ANCHOR="anchor";
    UID="descp"; # the unique descriptor of a image.
    META_DICT="meta_dict";
    @staticmethod
    def uid2str(uid):
        return "-".join([str(uid[k]) for k in uid]);
