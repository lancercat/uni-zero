import glob
import os
from collections import OrderedDict

from neko_2021_mjt.standardbench2_candidates.metas import METHOD_METAS, DEV_METAS
from neko_sdk.visualization.draw_curve import neko_make_curves_chart
from neko_sdk.visualization.draw_curve_area import draw_areas
from neko_sdk.visualization.neko_show_chart import neko_qt_chart_gui



import numpy as np
def make_datapack(xss,yss,names,colors,ltypes,nodes):
    axs=[];
    ays=[]
    nteds=[];
    for tid in range(len(colors)):
        nt = {};
        nt["x"] = xss[tid];
        nt["y"] = yss[tid];
        axs+=xss[tid];
        ays+=yss[tid];
        nt["n"] = names[tid];
        nt["c"] = colors[tid];
        nt["l"] = ltypes[tid];
        nt["s"]=nodes[tid];
        nteds.append(nt);
    ys = [];
    for nt in nteds:
        ys += nt["y"];
    xrange = np.min(axs), np.max(axs);
    yrange = np.min(ays), np.max(ays);
    return xrange, yrange, nteds;


def build_widgets(resd,method_metas,dev_metas,acr=True,vdsg=None):
    widgets = [];
    pitrs=OrderedDict();
    grps=[];
    datadict={
    };
    for ds in resd:
        if(vdsg is not None):
            widn=vdsg[ds][0];
            ptstyle=vdsg[ds][1];
        else:
            widn=ds;
            ptstyle="";
        if(widn not in datadict):
            datadict[widn] = [], [], [], [], [], [],[];
            grps.append(widn);

        xss, yss_acr, yss_ned, names, colors, ltypes,ptstyles=datadict[widn];
        for method in resd[ds]:
            if(method not in method_metas ):
                print("unknown", method);
                continue;
            color=method_metas[method]["color"];
            for mach in resd[ds][method]:
                if mach not in dev_metas:
                    continue;
                ltype=dev_metas[mach]["ltype"];
                xs=[];
                ys_acr=[];
                ys_ned=[];
                pitrs=pitrs | resd[ds][method][mach].keys();
                spitrs=sorted(list(pitrs))
                for i in spitrs:
                    if i in resd[ds][method][mach]:
                        xs.append(i);
                        ys_acr.append(resd[ds][method][mach][i ][0]);
                        try:
                            ys_ned.append(resd[ds][method][mach][i ][1]);
                        except:
                            pass;

                xss.append(xs);
                yss_acr.append(ys_acr);
                yss_ned.append(ys_ned);
                ltypes.append(ltype);
                colors.append(color);
                ptstyles.append(ptstyle);
                if("disp" not in method_metas[method]):
                    names.append(method + "@" + mach);
                else:
                    names.append(method_metas[method]["disp"] + "@" + mach);
    for widn in grps:
        xss, yss_acr, yss_ned, names, colors, ltypes,ptstyles=datadict[widn];
        if(len(xss)):
            if(acr):
                xrange, yrange, nteds=make_datapack(xss,yss_acr,names,colors,ltypes,ptstyles);
                Yname="Accuracy"
            else:
                xrange, yrange, nteds=make_datapack(xss,yss_ned,names,colors,ltypes,ptstyles);
                Yname="AR(1-ned)"

            widgets.append(neko_make_curves_chart(nteds,xrange,yrange,title=widn,Yname=Yname));
    return widgets
def draw(resd,path,method_metas,dev_metas,vdsg,acr=True):
    graphWidgets = build_widgets(resd,method_metas,dev_metas,acr,vdsg);
    mw=neko_qt_chart_gui.draw_multi(graphWidgets,path);
    if (path is not None):
        pix = mw.grab()
        pix.save(path)
    return mw;
def multi_run_figure(resd,path,method_metas,dev_metas,acr=True):
    pass;
    draw_areas();

if __name__ == '__main__':
    def register_method(log):
        with open(log, "r") as fp:
            lines = [i.strip() for i in fp];
        dsd = {};
        ds = "Meh";
        for i in lines:
            if i.find("starts") != -1:
                ds = i.rsplit(" ", 1)[0];
            if (i[:8] == "Accuracy"):
                dsd[ds] = [float(t.split(":")[1]) for t in i.split(",")];
        return dsd;


    def get_resd(root_dir, pat="*CE?"):
        logs = glob.glob(os.path.join(root_dir, pat, "TESTDAN.log"));

        mnames = [os.path.basename(os.path.dirname(i)) for i in logs];
        devs = [n.split("_", 1)[0] for n in mnames];
        itrs = [int(n.rsplit("_", 1)[1].replace("CE", "")) for n in mnames];
        keys = [n.split("_", 1)[1].rsplit("_", 1)[0] for n in mnames];

        adsd = {}
        for log, dev, itr, key in zip(logs, devs, itrs, keys):
            dsd = register_method(log);
            for k in dsd:
                if (k not in adsd):
                    adsd[k] = {};
                if (key not in adsd[k]):
                    adsd[k][key] = {}
                if dev not in adsd[k][key]:
                    adsd[k][key][dev] = {};
                adsd[k][key][dev][itr] = dsd[k];
        return adsd;

    root_dir="/home/lasercat/ssddata/what_is_hard";
    resd=get_resd(root_dir,"*mk5*CE?");
    draw(resd,"/run/media/lasercat/shared/cvpr22candidates_analysis/a.png",METHOD_METAS,DEV_METAS)
    # print(list());
