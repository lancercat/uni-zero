# define how a network input looks like in terms of shape
# you use this to init grid sampler targets
#
class neko_basic_im_dataspec:
    SIZE_hw="size_hw";
    NAME="name";

    @classmethod
    def get_default_template(cls, szhw,name):
        return {
            cls.SIZE_hw: szhw,
            cls.NAME:name
        };


