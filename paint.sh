#!/usr/bin/env bash
# Serve the browser/iPad CC/TD seg painter (cranial-up). Draw with the Apple Pencil
# in Safari on an iPad on the same WiFi; edits save to <stem>_edit.npz (same sidecar
# the napari editor + every viewer already read).
#   ./paint.sh cc_contexts/context4d_07_20_22_data_Baseline_Acq10   # one acq (all frames)
#   ./paint.sh cc_contexts/context4d_.../f3.npz                     # a single frame
#   ./paint.sh <input> --port 8000                                  # pick the port
# The command prints the http://<mac-lan-ip>:<port>/ URL to open on the iPad.
cd "$(dirname "$0")" || exit 1
PYTHONPATH=src exec python3 -m cc_reservoir.viz.ipad_paint "$@"
