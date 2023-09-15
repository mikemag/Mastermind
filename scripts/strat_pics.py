# Copyright (c) Michael M. Magruder (https://github.com/mikemag)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import subprocess


def render_graphs():
    strat_files = glob.glob("../results/*.gv")

    for fn in strat_files:
        jfn = fn + ".jpg"

        render = False
        if os.path.isfile(jfn):
            fn_stats = os.stat(fn)
            jfn_stats = os.stat(jfn)
            if fn_stats.st_mtime > jfn_stats.st_mtime:
                render = True
        else:
            render = True

        if render:
            cmd = ["twopi", "-Tjpg", "-O", fn]
            print(cmd)
            subprocess.run(cmd)


if __name__ == "__main__":
    render_graphs()
