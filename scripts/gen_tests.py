# Copyright (c) Michael M. Magruder (https://github.com/mikemag)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
from collections import OrderedDict

from sortedcontainers import SortedDict


def load_json(filename, results, initial_guesses):
    with open(os.path.join(filename), "r") as f:
        r = json.load(f)
        runs = [x["run"] for x in r if "run" in x]

        for run in runs:
            if "Total Turns" not in run:
                continue

            ar = results.setdefault(run["Strategy"], SortedDict())
            pr = ar.setdefault(int(run["Pin Count"]), SortedDict())
            gr = pr.setdefault(
                int(run["Color Count"]),
                {
                    "total_turns": 999_999_999_999_999,
                    "max_turns": 9999,
                    "sample_games": [],
                },
            )

            total_turns = int(run["Total Turns"])
            max_turns = int(run["Max Turns"])

            if total_turns < gr["total_turns"]:
                gr["total_turns"] = total_turns
                gr["max_turns"] = max_turns
                gr[
                    "desc"
                ] = f"{run['Strategy']} {run['Pin Count']}p{run['Color Count']}c"

            if "Sample Game" in run and total_turns == gr["total_turns"]:
                sg = run["Sample Game"]
                if sg not in gr["sample_games"]:
                    ig = (
                        initial_guesses.get(run["Strategy"], {})
                        .get(str(run["Pin Count"]), {})
                        .get(str(run["Color Count"]), {})
                        .get("best_avg", [])
                    )
                    if ig and ig[2] == sg[0]:
                        gr["sample_games"].append(sg)


def process_results():
    with open("../docs/initial_guesses/preset_initial_guesses.json", "r") as f:
        initial_guesses = json.load(f)

    results = OrderedDict()
    result_files = []
    result_files.extend(glob.glob("../results/**/*.json", recursive=True))
    # result_files.extend(glob.glob("../*_ig_*.json", recursive=True))

    for f in result_files:
        print(f)
        load_json(f, results, initial_guesses)

    # Add in Knuth's example for 4p6c
    sgs = results.get("Knuth", {}).get(4, {}).get(6, {}).get("sample_games", [])
    if sgs:
        sgs.append(["1122", "1344", "3526", "1462", "3632"])

    with open("../valid_solutions.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    process_results()
