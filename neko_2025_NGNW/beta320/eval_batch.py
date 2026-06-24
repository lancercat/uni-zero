import os

import tqdm

import os.path

import torch
import re

def add_one(srcdst,root,mname,runname,id=0,things=["ACR","Recall","Precision","FScore"]):
    mf=torch.load(os.path.join(root,runname+".pt"),weights_only=False);
    for itrk in mf:
        for pk in mf[itrk][id]:
            runres=mf[itrk][id][pk];
            if pk not in srcdst:
                srcdst[pk]={};
            if(mname not in srcdst[pk]):
                srcdst[pk][mname]={};
                for thing in things:
                    srcdst[pk][mname][thing]={};
            rk=runname+itrk;
            for thing in things:
                if(thing in runres):
                    srcdst[pk][mname][thing][rk]=runres[thing];
    return srcdst;


def generate_stats(rd):
    stats = {}
    for pk, models in rd.items():
        stats[pk] = {}
        for mname, metrics in models.items():
            stats[pk][mname] = {}
            for metric_name, runs in metrics.items():
                # Extract values from the rk:value dictionary
                values = list(runs.values())

                if len(values) > 0:
                    # Convert to tensor for easy math
                    v_tensor = torch.tensor(values, dtype=torch.float32)*100.0

                    stats[pk][mname][metric_name] = {
                        "avg": v_tensor.mean().item(),
                        "std": v_tensor.std().item() if len(values) > 1 else 0.0
                    }
    return stats


def export_stats_commands(sd, pk_latex_map, model_names):
    """Generates LaTeX commands for Averages and Standard Deviations (Summary Data)."""
    lines = ["% ============================================================",
             "% SUMMARY STATISTICS (AVERAGES & STD DEVS)",
             "% ============================================================\n"]
    metrics = ["ACR", "Recall", "Precision", "FScore"]

    for mname in model_names:
        lines.append(f"% --- Method: {mname} ---")
        for pk_key, task_suffix in pk_latex_map.items():
            if pk_key not in sd: continue

            model_avg_data = sd[pk_key].get(mname, {})
            for metric in metrics:
                metric_label = metric.capitalize() if metric != "ACR" else "Acr"
                base_cmd = f"N{mname}{task_suffix}{metric_label}"

                stats = model_avg_data.get(metric, {})
                avg = stats.get("avg", "-")
                std = stats.get("std", "-")

                # Format as float only if the value is a number (int or float)
                avg_str = f"{avg:.2f}" if isinstance(avg, (int, float)) else avg
                std_str = f"{std:.2f}" if isinstance(std, (int, float)) else std

                lines.append(f"\\newcommand{{\\{base_cmd}Avg}}{{{avg_str}}}")
                lines.append(f"\\newcommand{{\\{base_cmd}Std}}{{{std_str}}}")
        lines.append("")  # Spacer
    return "\n".join(lines)


def get_roman(n):
    """Converts 1->I, 2->II, etc."""
    mapping = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
    return mapping.get(int(n), str(n))


def extract_run_id(run_full_name):
    """
    Extracts the run number from the suffix '-run[i]'.
    Defaults to 1 if no suffix is found.
    """
    match = re.search(r'-run(\d+)', run_full_name)
    if match:
        return int(match.group(1))
    return 1


def export_raw_commands(rd, pk_latex_map, itrs_latex_map, model_names):
    lines = ["% ============================================================",
             "% RAW ITERATION DATA (Extracted from -run suffix)",
             "% ============================================================\n"]
    metrics = ["ACR", "Recall", "Precision", "FScore"]

    for mname in model_names:
        lines.append(f"% --- Method: {mname} ---")
        for pk_key, task_suffix in pk_latex_map.items():
            if pk_key not in rd: continue

            model_raw_data = rd[pk_key].get(mname, {})
            for metric in metrics:
                metric_label = metric.capitalize() if metric != "ACR" else "Acr"
                base_cmd = f"N{mname}{task_suffix}{metric_label}"

                run_values = model_raw_data.get(metric, {})
                for run_full_name, val in run_values.items():
                    # 1. Determine Run ID from suffix
                    rid = extract_run_id(run_full_name)
                    run_rom = f"R{get_roman(rid)}"

                    # 2. Match Iteration suffix
                    for iter_key, iter_rom in itrs_latex_map.items():
                        if run_full_name.endswith(iter_key):
                            val_s = f"{val*100.0:.2f}"
                            # Example: \NoursFslValAcrVIIIRI
                            lines.append(f"\\newcommand{{\\{base_cmd}{iter_rom}{run_rom}}}{{{val_s}}}")
                            break
        lines.append("")
    return "\n".join(lines)


def collect_results(root,resd):
    rd={};
    for mname in resd:
        for rname in resd[mname]:
            add_one(rd,root,mname,rname,0);

    return rd;




class executer:
    def __init__(this):
        this.dir=os.path.abspath(os.curdir);
        this.pythonpath=os.path.join(this.dir,"../../");
        print(this.pythonpath);
        this.itrs_map = {
            "_E1_I20000": "VIII",
            "_E1_I40000": "X",
            "_E2": "XII",
            "_E2_I20000": "XIV"
        }
    def execute(this,expname):
        os.chdir(this.dir);
        print(this.dir);
        os.chdir(expname);
        os.system("PYTHONPATH="+this.pythonpath+" python eval.py");

    def eval_batch(this,exptree):
        for protocol in  tqdm.tqdm(exptree):
            for mname in tqdm.tqdm(exptree[protocol]):
                for run in  tqdm.tqdm(exptree[protocol][mname]):
                    this.execute(run);

    def generate_latex_report(self,rdir, exptree,proto_pk_map, output_file="results.tex"):
        """
        Gathers results from the exptree, calculates stats, and exports LaTeX.
        """
        pk_map = {k: k for k in exptree.keys()}  # Maps Protocol keys (Indc, Ostr)
        all_stats_output = []
        all_raw_output = []

        # Process each protocol (Indc, Ostr)
        for protocol, models in exptree.items():
            # collect_results expects {mname: [run_list]}
            # It returns rd[pk][mname][metric][run_full_name]
            rd = collect_results(rdir, models)
            sd = generate_stats(rd)

            # Map specific for this protocol iteration
            # proto_pk_map = {protocol: pk_map.get(protocol, protocol)}
            model_names = list(models.keys())

            # 1. Generate Summary (Avg/Std)
            all_stats_output.append(export_stats_commands(sd, proto_pk_map, model_names))

            # 2. Generate Raw Iteration Data using ITRS_LATEX logic
            all_raw_output.append(export_raw_commands(rd, proto_pk_map, self.itrs_map, model_names))

        # Write final .tex file
        with open(output_file, "w") as f:
            f.write("% Generated Experiment Macros\n")
            f.write("\n".join(all_stats_output))
            f.write("\n\n% --- Raw Run Data ---\n\n")
            f.write("\n".join(all_raw_output))

        print(f"Done! Macros saved to {output_file}")



    pass;
if __name__ == '__main__':
    PD={'jpn_hori_gzsl-task_performance':"GZSLJP",
        'kr_hori_gzsl-task_performance':"GZSLKR",
        'syn_yi_hori_gzsl-task_performance': "GZSLSYi",
        "istrval_bengali-task_performance": "GZSLVBE",
        "istrval_gujarati-task_performance": "GZSLVGU",
        "istrval_hindi-task_performance": "CVHI",
        "istrval_punjabi-task_performance": "CVPU",
        "istrval_kannada-task_performance": "CVKA",
        "istrval_tamil-task_performance": "CVTE",
        "istrval_telugu-task_performance": "CVTA",
        "istrval_odia-task_performance": "CVOD",
        "istrval_marathi-task_performance": "CVMAR",
        "istrval_malayalam-task_performance": "CVMAL",
        "istrtest_bengali-task_performance": "GZSLTBE",
        "istrtest_gujarati-task_performance": "GZSLTGU",
        "istrtest_hindi-task_performance": "CTHI",
        "istrtest_punjabi-task_performance": "CTPU",
        "istrtest_kannada-task_performance": "CTKA",
        "istrtest_tamil-task_performance": "CTTE",
        "istrtest_telugu-task_performance": "CTTA",
        "istrtest_odia-task_performance": "CTOD",
        "istrtest_marathi-task_performance": "CTMAR",
        "istrtest_malayalam-task_performance": "CTMAL",
        }
    STRA={
        "Indc":{
            "CamCtc" : ["b1h1NF-1024-indc2-cntralgn-lcam-ctc-32-128","b1h1NF-1024-indc2-cntralgn-lcam-ctc-32-128-run2"],
            "CamLpos": ["b1h1NF-1024-indc2-cntralgn-lcam-lposReal-32-128","b1h1NF-1024-indc2-cntralgn-lcam-lposReal-32-128-run2"],
            "SqzCtc": ["b1h1NF-1024-indc2-cntralgn-sqz-lctc-32-128","b1h1NF-1024-indc2-cntralgn-sqz-lctc-32-128-run2"]
        },
        "Ostr": {
            "CamCtc" :["b1NF-1024-ostr-cntralgn-lcam-lpos-32-128","b1NF-1024-ostr-cntralgn-lcam-lpos-32-128-run2"],
            "CamLpos": ["b2NF-1024-ostr-cntralgn-lcam-lposReal-32-128",
                        "b2NF-1024-ostr-cntralgn-lcam-lposReal-32-128-run2"],
            "SqzCtc": ["b1dNF-1024-ostr-cntralgn-sqz-lctc-32-128","b1dNF-1024-ostr-cntralgn-sqz-lctc-32-128-run2"]
        }
    }
    CFLA={
        "Indc": {
            "BigBuc": ["b1h4NF-1024-indc-cntralgn-lcam-ctc-32-128"],
            "ConflCamCtc": ["b1h1NF-1024-indc2-cntralgn-lcam-ctc-32-128"],
        }
    }
    LSCTA={
         "Ostr": {
        "Xbatch": ["b1NFXB4-1024-ostr-cntralgn-lcam-lpos-32-128", "b1NFXB4-1024-ostr-cntralgn-lcam-lpos-32-128-run2"],
        "ShufSA": ["b1NF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT","b1NF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT-run2"],
            "FRC": ["b1iNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctFRC_RealNI64",
                   "b1iNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctFRC_RealNI64-run2"],
        "ShufFRC": ["b1jNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT-lsctFRC_RealNI64","b1jNF-1024-ostr-cntralgn-lcam-lpos-32-128-lsctSHSTT-lsctFRC_RealNI64-run2"],
    },
        "indc": {
            "ShufSA": ["b1h2NF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT",
                       "b1h2NF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT-run2"],

            "FRC": ["b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctFRC_RealNI64",
                          "b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctFRC_RealNI64-run2"],
            "ShufFRC": ["b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT-lsctFRC_RealNI64",
                          "b1jNF-1024-indc2-cntralgn-lcam-ctc-32-128-lsctSASTT-lsctFRC_RealNI64-run2"],
        }
    }

    e=executer();
    e.eval_batch(STRA);
    e.generate_latex_report("/run/media/lasercat/320-eccv/results/hydra_results_eccv/", STRA, PD,
                            "/run/media/lasercat/320-eccv/results/hydra_results_eccv/stra.tex")
    e.eval_batch(CFLA);
    e.generate_latex_report("/run/media/lasercat/320-eccv/results/hydra_results_eccv/", CFLA, PD,
                            "/run/media/lasercat/320-eccv/results/hydra_results_eccv/cfla.tex")
    e.eval_batch(LSCTA);
    e.generate_latex_report("/run/media/lasercat/320-eccv/results/hydra_results_eccv/",LSCTA,PD,"/run/media/lasercat/320-eccv/results/hydra_results_eccv/lscta.tex")