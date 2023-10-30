# Copyright (c) Michael M. Magruder (https://github.com/mikemag)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from collections import OrderedDict


def load_json(filename, results):
    with open(filename, "r") as f:
        r = json.load(f)
        sysinfo = [x["system_specs"] for x in r if "system_specs" in x][0]
        runs = [x["run"] for x in r if "run" in x]

        for run in runs:
            ar = results.setdefault(run["Strategy"], {})
            pr = ar.setdefault(int(run["Pin Count"]), {})
            gr = pr.setdefault(
                int(run["Color Count"]),
                {
                    "best": {
                        "avg": 9999.9,
                        "ig": "",
                        "max": 9999,
                        "time": 99999.9,
                        "scores": 999_999_999_999_999,
                    },
                },
            )

            d = {
                "ig": run["Initial Guess"],
                # nb, old results had lower precision.
                "avg": round(float(run["Average Turns"]), 5),
                "max": int(run["Max Turns"]),
                "time": float(run["Elapsed (s)"]),
                "scores": int(run["Scores"]),
                "sysinfo": sysinfo,
            }

            if d["avg"] < gr["best"]["avg"] or (
                d["avg"] == gr["best"]["avg"]
                and run["Solver"] == "CUDA"
                and d["time"] < gr["best"]["time"]
            ):
                gr["best"] = d


def build_system_list(results):
    systems = {}
    for a, pd in results.items():
        for p, cd in pd.items():
            for c, gd in cd.items():
                si = gd["best"]["sysinfo"]
                sys = f"{si['GPU Name']}, {si['HW CPU Brand String']}, CUDA Toolkit {si['GPU CUDA Runtime Version']} on {si['OS Product Version']}, {si['OS Kernel Version']}"
                systems.setdefault(sys, 0)
                systems[sys] += 1
                gd["best"]["sys_str"] = sys

    system_names = [
        k for k, v in sorted(systems.items(), key=lambda item: item[1], reverse=True)
    ]
    return system_names


def process_results(f, results, systems, metric, metric_format, header, desc):
    f.write(f"## {header}\n\n")

    if desc:
        f.write(desc)
        f.write("\n\n")

    for a, pd in results.items():
        f.write(f"### {a}\n\n")
        f.write("|".join(["", " ", *[str(c) + "c" for c in range(2, 16)], ""]))
        f.write("\n")
        f.write("".join(["|---:" * 15, "|"]))
        f.write("\n")

        for p in range(2, 9):
            f.write("|" + str(p) + "p")
            if p in pd:
                cd = pd[p]
                for c in range(2, 16):
                    f.write("|")
                    if c in cd:
                        gd = cd[c]
                        ba = gd["best"]
                        # todo: lots of special cases for time... annoying.
                        if metric == "time":
                            if ba[metric] < 100:
                                f.write(metric_format.format(ba[metric]))
                            else:
                                f.write("{:,d}".format(int(ba[metric])))
                            sys_num = systems.index(ba["sys_str"]) + 1
                            if sys_num > 1:
                                f.write(f"<sup>({sys_num})</sup>")
                        else:
                            f.write(metric_format.format(ba[metric]))
            f.write("|\n")
        f.write("\n")


def grouped_by_game(f, results, systems, metrics):
    f.write("## Grouped by Game\n\n")
    f.write("The same metrics grouped by game, to easily compare algorithms.\n")

    algos = results.keys()
    min_pins = min([min(results[a].keys()) for a in results.keys()])
    max_pins = max([max(results[a].keys()) for a in results.keys()])
    min_colors = min(
        [min(results[a][p].keys()) for a in results.keys() for p in results[a].keys()]
    )
    max_colors = max(
        [max(results[a][p].keys()) for a in results.keys() for p in results[a].keys()]
    )

    for p in range(min_pins, max_pins + 1):
        if any([True for a in algos if p in results[a]]):
            for c in range(min_colors, max_colors + 1):
                if any([True for a in algos if p in results[a] and c in results[a][p]]):
                    f.write(f"\n### {p}p{c}c\n\n")
                    f.write("| |" + "|".join([m["name"] for m in metrics]) + "|\n")
                    f.write("|:---|" + "---:|" * len(metrics) + "\n")
                    for a in algos:
                        if p not in results[a] or c not in results[a][p]:
                            continue
                        f.write("|")
                        f.write(f"{a}|")
                        for m in metrics:
                            ba = results[a][p][c]["best"]
                            f.write(m["fmt"].format(ba[m["metric"]]))
                            f.write(" |")
                        f.write("\n")


if __name__ == "__main__":
    results = OrderedDict()
    result_files = []
    # result_files.extend(glob.glob("../results/2022_i7-10700K_CUDA_3070_ubuntu22/*.json"))
    # result_files.extend(glob.glob("../results/2023_GCE_Various/*.json"))

    # These are the only results using the CE opt at this time, only compare those.
    result_files.extend(glob.glob("../results/2023_4090/*.json"))

    for f in result_files:
        load_json(f, results)

    systems = build_system_list(results)

    with open("../results/README.md", "w") as f:
        f.write("# Results\n\n")
        f.write(
            "All results obtained using SolverCUDA from one of the following "
            "systems:\n\n"
        )
        f.write("\n".join([f"{i + 1}. {s}" for i, s in enumerate(systems)]))
        f.write("\n\n")
        f.write(
            "I only show results for games where I've determined the best "
            "starting guess. This takes a long time for larger games, thus the "
            "tables are only partially filled.\n\n"
        )
        f.write(
            "All runs have all gameplay optimizations enabled, like small fully "
            "discriminating sets, case equivalence, etc. "
            "See [Mastermind on the GPU](../docs/Mastermind_on_the_GPU.md) for details. "
            "Note: for the CUDA impl, the case equivalence opt applies when $c^p > 400000$."
            "\n\n"
        )
        f.write("*Times reported are from the first system unless otherwise marked.* ")
        f.write("*Raw data is in the .json files in subdirectories here.*\n\n")

        metrics = [
            {
                "name": "Avg",
                "metric": "avg",
                "fmt": "{:,.4f}",
                "header": "Average turns over all games",
                "desc": "",
            },
            {
                "name": "Max",
                "metric": "max",
                "fmt": "{:,d}",
                "header": "Max turns over all games",
                "desc": "",
            },
            {
                "name": "Time (s)",
                "metric": "time",
                "fmt": "{:,.3f}",
                "header": "Run time",
                "desc": "All times in seconds.",
            },
            {
                "name": "Initial guess",
                "metric": "ig",
                "fmt": "{}",
                "header": "Initial guess",
                "desc": "These were determined by running all unique initial guesses "
                "and selecting the one with the lowest average turns.",
            },
            {
                "name": "Scores",
                "metric": "scores",
                "fmt": "{:,d}",
                "header": "Codeword scores performed",
                "desc": "",
            },
        ]

        for m in metrics:
            process_results(
                f, results, systems, m["metric"], m["fmt"], m["header"], m["desc"]
            )

        grouped_by_game(f, results, systems, metrics)
