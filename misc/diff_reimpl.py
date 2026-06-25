import re


def parse_latex_macros(file_path):
    """Parses the LaTeX macro file into a structured dictionary."""
    macro_pattern = re.compile(r'\\newcommand\{\\N([A-Za-z0-9]+)\}\{(.*?)\}')
    data = {"summary": {}, "raw_runs": {}}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = macro_pattern.findall(content)
    for macro_name, value in matches:
        try:
            cleaned_value = float(value)
        except ValueError:
            cleaned_value = value.strip()

        run_match = re.search(r'(XRI|XIIRI|XIVRI)$', macro_name)

        if run_match:
            run_type = run_match.group(1)
            metric_base = macro_name[:-len(run_type)]
            if metric_base not in data["raw_runs"]:
                data["raw_runs"][metric_base] = {}
            data["raw_runs"][metric_base][run_type] = cleaned_value
        else:
            if macro_name.endswith("Avg"):
                metric_base = macro_name[:-3]
                stat_type = "Avg"
            elif macro_name.endswith("Std"):
                metric_base = macro_name[:-3]
                stat_type = "Std"
            else:
                metric_base = macro_name
                stat_type = "Value"

            if metric_base not in data["summary"]:
                data["summary"][metric_base] = {}
            data["summary"][metric_base][stat_type] = cleaned_value

    return data


def diff_reimpl(tex_path1, tex_path2):
    """
    Compares two LaTeX macro files and returns a dictionary containing
    the absolute differences for matching numeric keys.
    """
    dict1 = parse_latex_macros(tex_path1)
    dict2 = parse_latex_macros(tex_path2)

    diffdict = {"summary": {}, "raw_runs": {}}

    # Helper to calculate differences between two nested sub-dictionaries
    def calculate_sub_diff(subdict1, subdict2):
        sub_diff = {}
        # Only compare keys present in both files to avoid KeyErrors
        shared_metrics = set(subdict1.keys()) & set(subdict2.keys())

        for metric in shared_metrics:
            shared_subkeys = set(subdict1[metric].keys()) & set(subdict2[metric].keys())
            metric_diffs = {}

            for subkey in shared_subkeys:
                v1 = subdict1[metric][subkey]
                v2 = subdict2[metric][subkey]

                # Check that both values are numeric (floats/ints) and not strings like '-'
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    metric_diffs[subkey] = round(abs(v1 - v2), 4)

            if metric_diffs:
                sub_diff[metric] = metric_diffs

        return sub_diff

    diffdict["summary"] = calculate_sub_diff(dict1["summary"], dict2["summary"])
    diffdict["raw_runs"] = calculate_sub_diff(dict1["raw_runs"], dict2["raw_runs"])

    return diffdict


def thresh_diff(diffdict, thresh):
    """
    Filters a diffdict, returning a new dictionary containing only the
    metrics where at least one subkey variance meets or exceeds the threshold.
    """
    filtered_dict = {"summary": {}, "raw_runs": {}}

    for category in ["summary", "raw_runs"]:
        for metric, subkeys in diffdict[category].items():
            # Keep the metric if any of its values (e.g., Avg, Std, or XRI) >= thresh
            if any(value >= thresh for value in subkeys.values()):
                filtered_dict[category][metric] = subkeys

    return filtered_dict

diff_report = diff_reimpl( '/run/media/lasercat/data/cat/Object320-EX/results/stra.tex','/run/media/lasercat/data/cat/Object320-EX/results/strb.tex')

# 2. Filter for significant performance shifts (e.g., delta >= 1.0)
significant_changes = thresh_diff(diff_report, 0.1)
print(significant_changes)