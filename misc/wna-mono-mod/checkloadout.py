import shutil

regdict={
    "base":[
        "project_310_v6SF-routing/rule_based-v6S-AAF",
        "project_310_v6SF-routing-r2/rule_based-v6S-AAF-run2"
    ],
    "base-S":[
        "project310_v6SF_stability-re/rule_based-v6S-AAF-ohem01E",
        "project310_v6SF_stability-re2/rule_based-v6S-AAF-ohem01E-run2"
    ],
    "R1":[
        "project_310_v6SF-routing/pgroute_nd_only-v6S-tiny-AAF",
        "project_310_v6SF-routing-r2/pgroute_nd_only-v6S-tiny-AAF-run2",
    ],
    "R1-S": [
        "project_310_v6SF-routing-stablized/pgroute_nd_only-v6S-tiny-AAF-01E",
        "project_310_v6SF-routing-stablized-r2/pgroute_nd_only-v6S-tiny-AAF-01E-run2"
    ],
    "R2": [
        "project_310_v6SF-routing/pgroute_nd_only-v6S-tiny-AAF-nedmix",
        "project_310_v6SF-routing-r2/pgroute_nd_only-v6S-tiny-AAF-nedmix-run2"
    ],
    "R2-S":[
        "project_310_v6SF-routing-stablized/pgroute_nd_only-v6S-tiny-AAF-nedmix-01E",
        "project_310_v6SF-routing-stablized-r2/pgroute_nd_only-v6S-tiny-AAF-nedmix-01E-run2"
    ],
    "R3": [
       "project_310_v6SF-routing/aroute_nd_only-v6S-tiny-AAF",
       "project_310_v6SF-routing-r2/aroute_nd_only-v6S-tiny-AAF-run2"
    ],
    "R3-S": [
        "project_310_v6SF-routing-stablized/aroute_nd_only-v6S-tiny-AAF-01E",
        "project_310_v6SF-routing-stablized-r2/aroute_nd_only-v6S-tiny-AAF-01E-run2"
    ],
    "WNA":[
        "project_310_v6SF-routing/aroute_nd_only-v6S-tiny-nedmix-AAF",
        "project_310_v6SF-routing-r2/aroute_nd_only-v6S-tiny-nedmix-AAF-run2"
    ],
    "WNA-S":[
        "project310_v6SF_stability-re/aroute_nd_only-v6S-tiny-nedmix-AAF-ohem01E",
        "project310_v6SF_stability-re2/aroute_nd_only-v6S-tiny-nedmix-AAF-ohem01E-run2"
    ],
    "WNA-XL-S":[
        "project310_v6SF_stability-re/aroute_nd_only-v6S-tiny-nedmix-AAF-ohem01E",
        "project310_v6SF_stability-re2/aroute_nd_only-v6S-tiny-nedmix-AAF-ohem01E-run2"
    ],
    "WNA-SA":[
        "project310_v6SF_stability-re/aroute_nd_only-v6S-tiny-nedmix-AAF-ohem01",
        "project310_v6SF_stability-re2/aroute_nd_only-v6S-tiny-nedmix-AAF-ohem01-run2"
    ]
}
import os;
root = os.getcwd();
os.environ["PYTHONPATH"] = root;
for k in regdict:
    for runid in range(len(regdict[k])):
        run=regdict[k][runid];
        l=os.path.join(root,"object310-rel",run,"loadout.py");
        shutil.copy(l,os.path.join(root,k+"-loadout"+str(runid)));
    os.system("git diff "+os.path.join(root,k+"-loadout0")+ " "+os.path.join(root,k+"-loadout1")+ "  >> alldiff.diff");
