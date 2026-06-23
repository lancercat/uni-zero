import os.path

from osocrNG.athena.common.quicklmdb import quick_lmdb
from osocrNG.athena.common.quickptg1 import prepare_pt


def quick_dataset(langroot):
    prepare_pt(langroot);
    quick_lmdb(langroot,os.path.join(langroot,"lmdb"),"None");

