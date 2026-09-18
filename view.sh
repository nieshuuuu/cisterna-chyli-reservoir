#!/usr/bin/env bash
# Open the whole-body MIP panel on the built CC/TD contexts.
#   ./view.sh                         # acq dropdown over cc_contexts/  (s=our seg, l=lab compare)
#   ./view.sh cc_contexts/context4d_07_20_22_data_Angiotensin_Acq8   # one acq
#   ./view.sh <ctx> --shot out.png --lab                             # headless grab
# Defaults to cc_contexts/ when no path is given.
cd "$(dirname "$0")" || exit 1
PYTHONPATH=src exec python3 -m cc_reservoir.viz.mip_panel "${@:-cc_contexts/}"
