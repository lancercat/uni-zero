class neko_diskcol_template:
    RAW_IM_MAX_AREA="raw_max_size";  # incase huge image hog ram.
    THUMB_SIZE = "thumbnail_size";
    THUMB_PAD = "thumbnail_pad";
    @classmethod
    def get_default_diskcol_template(cls):
        return {
            neko_diskcol_template.RAW_IM_MAX_AREA: 128 * 128, # this exist to save your ram. You don't want uncompressed 4k images in your data queue.
            neko_diskcol_template.THUMB_PAD: [0, 0],
            neko_diskcol_template.THUMB_SIZE: [48, 48]
        };

