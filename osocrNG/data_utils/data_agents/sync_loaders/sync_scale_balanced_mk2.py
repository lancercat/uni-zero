import copy
import random

import torch
import tqdm
from multiprocessing import Queue as mpQueue
import multiprocessing

from osocrNG.data_utils.data_agents.sync_loaders.abstract_sync_im_text_loader import neko_abstract_im_text_sync_random_multidb
from neko_sdk.neko_framework_NG.workspace import neko_workspace, neko_environment
from osocrNG.data_utils.data_agents.multilmdb_dispatching_agent import neko_fetching_and_dispatching_servant
from osocrNG.data_utils.raw_names import raw_data_item_names as RN
from osocrNG.data_utils.raw_names import basic_data_info_mk2 as BIN2

from neko_sdk.log import fatal,warn
import pickle
from neko_sdk.log import info
