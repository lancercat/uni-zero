import os
import torch

def panic(workspace, environment=None):
    os.makedirs("badmods",exist_ok=True);
    if (environment):
        environment.modset.save_as_necessary("bad_mods", workspace.epoch_idx, workspace.batch_idx);
    torch.save(workspace, "offending_ws.pt");