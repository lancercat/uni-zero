# why training meta and testing meta?
# reducing human errors ofc!
import os.path


class training_metas:
    ENCH="ENG-CHS-3755";
    EN="MJST-nosym";
    @ classmethod
    def get_dict(cls,root):
        return {
            cls.ENCH: os.path.join(root, "dictsv2", "dab3791MC"),
            cls.EN: os.path.join(root, "dictsv2", "dab62cased")
        }

class testing_metas:
    EN="MJST-nosym";
    FUDANDICT="FUDAN";
    JPNGZSL="JPN-GZSL";
    JPNOSR = "JPN-OSR";
    JPNGOSR="JPN-GOSR";
    JPNOSTR = "JPN-OSTR";

    MOOSTRGZSL="JPNHV-GZSL";
    MOOSTROSR = "JPNHV-OSR";
    MOOSTRGOSR="JPNHV-GOSR";
    MOOSTROSTR = "JPNHV-OSTR";

    KRGZSL="KR-GZSL";

