class neko_abstract_data_source_NG:
    def setup(this,para):
        pass;

    ### The descp is a dict, so that is why we cannot use the dataset class here.
    def fetch_item(this, descp):
        pass;
    def disarm(this):
        this.nSamples = 0;

    def __init__(this,para):
        this.nSamples=0;
        this.setup(para);
    def all_valid_indexes(this):
        return [];

