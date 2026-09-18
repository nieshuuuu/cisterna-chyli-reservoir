#!/bin/sh
# CC/TD painter launcher (Mac / Linux).  Run:  sh start_cc_painter.sh
#
# On Windows use Paint.bat instead: it syncs your masks to the share before and after the
# session. There is no PowerShell here, so THIS script only opens the painter -- syncing is
# still yours to do, and it is the step people forget.
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="$PWD/software/src"

python3 -c 'import numpy, PIL' 2>/dev/null || {
    echo "Missing Python packages. Please run:  pip3 install numpy pillow"
    exit 1
}

SHARE_DATA="/Volumes/Molloilab/ImageData/Lymph_Studies/Cisterna_Chyli_and_Thoracic_Duct_Segmentation/2_CT_volumes_and_painter/data_fullres"

echo "============================================================"
echo "  CC/TD painter - full-resolution, segmented acquisitions"
echo "  1) Wait for the  http://...:8778/  line below."
echo "  2) On the iPad (same Wi-Fi) open that URL in Safari."
echo "     On this computer:  http://127.0.0.1:8778/"
echo "  Keep this terminal open while painting; Ctrl-C to stop."
echo "------------------------------------------------------------"
if [ "$PWD/data_fullres" != "$SHARE_DATA" ]; then
    echo "  THIS IS A LOCAL COPY. Your masks are saved to THIS machine"
    echo "  only. Nothing carries them to the share automatically on"
    echo "  Mac/Linux. When you finish, copy them up yourself:"
    echo ""
    echo "    rsync -av --include='*/' --include='f*_edit.npz' --exclude='*' \\"
    echo "      \"\$PWD/data_fullres/\" \\"
    echo "      \"$SHARE_DATA/\""
    echo ""
    echo "  Check first that nobody else edited the same frames."
    echo "------------------------------------------------------------"
fi
python3 -m cc_reservoir.viz.ipad_paint "$PWD/data_fullres/context4d_4_13_23_data_Acq01_Acq01" --port 8778
