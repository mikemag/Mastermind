# Copyright (c) Michael M. Magruder (https://github.com/mikemag)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob


def fixup_broken_files():
    result_files = glob.glob("../results/2023_GCE_Various/*.json")

    for filename in result_files:
        print(filename)
        with open(filename, "r") as f:
            lines = f.readlines()
        ll = lines[-1].strip()
        if len(ll) == 0 or ll[-1] != "]":
            with open(filename, "a") as f:
                f.write("]\n")


if __name__ == "__main__":
    fixup_broken_files()
