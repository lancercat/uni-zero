import copy
def make_deploy_anchor(module_anchor_config):
    flatten = {"names": [],"NEP_is_flatten_NEP":True};
    if("NEP_is_flatten_NEP" in module_anchor_config):
        if(module_anchor_config["NEP_is_flatten_NEP"]):
            return module_anchor_config; # like hell we will flatten twice. Do NOT rabbit down.
    for a in module_anchor_config["names"]:
        flatten["names"].append(a);
        flatten[a] = copy.copy(module_anchor_config[a]);
        if ("redundency" in module_anchor_config[a]):
            for replica_id in range(module_anchor_config[a]["redundency"]):
                flatten[a + "_" + str(replica_id)] = copy.copy(module_anchor_config[a]);
    return flatten;