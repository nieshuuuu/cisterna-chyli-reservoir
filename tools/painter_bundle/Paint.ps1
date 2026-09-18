# Paint.ps1 -- the whole painting session, one command.
#
#   1. push up anything left pending from last time
#   2. pull down the current painter code and everyone else's masks
#   3. open the painter on your LOCAL copy (fast)
#   4. when you stop it, push your masks back up
#
# The point is that steps 1, 2 and 4 are not yours to remember. The old workflow needed
# three separate actions and an acquisition (8_31_22_Acq2, six frames) was lost because
# one of them was skipped.
#
# Nothing is destroyed if this is killed rather than closed cleanly: the painter writes
# each Save straight to local disk, and step 1 of the NEXT run pushes whatever is pending.
#
# Usage:  Paint.bat            (double-click)
#         powershell -ExecutionPolicy Bypass -File .\Paint.ps1 -Port 8778
#         ... -Local D:\my_painter_copy    put the working copy somewhere else
#         ... -SyncOnly                    sync both ways, do not open the painter
#         ... -Yes                         do not ask before the first-run copy

param(
    [string] $Share,
    [string] $Local = 'C:\Storage\CisternaChyli\painter_work',
    [int]    $Port = 8778,
    [switch] $SyncOnly,
    [switch] $Yes
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot

function Say($m, $c = 'Gray') { Write-Host $m -ForegroundColor $c }
function Rule { Write-Host ('-' * 62) -ForegroundColor DarkGray }

if (-not $Share) {
    $candidates = @(
        '\\160.87.12.113\Molloilab\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation\2_CT_volumes_and_painter',
        'Z:\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation\2_CT_volumes_and_painter'
    )
    foreach ($c in $candidates) { if (Test-Path -LiteralPath $c) { $Share = $c; break } }
}
if (-not $Share -or -not (Test-Path -LiteralPath $Share)) {
    Say 'Cannot reach the lab share.' 'Red'
    Say 'Connect to the network (or map Z:) and run this again.' 'Red'
    exit 1
}

Write-Host ''
Write-Host '==============================================================' -ForegroundColor Cyan
Write-Host '  CC / TD painter' -ForegroundColor Cyan
Write-Host '==============================================================' -ForegroundColor Cyan
Say "  share:  $Share"
Say "  local:  $Local"
Write-Host ''

# ---------------------------------------------------------- first-run bootstrap --
if (-not (Test-Path -LiteralPath (Join-Path $Local 'data_fullres'))) {
    $size = 16
    Say 'This machine has no local copy yet.' 'Yellow'
    Say "Painting across the network is slow, so the CT volumes are copied here once:" 'Yellow'
    Say "    $Local" 'Yellow'
    Say "That is about $size GB and takes a while. Afterwards only masks move, in seconds." 'Yellow'
    Write-Host ''
    if (-not $Yes) {
        $answer = Read-Host 'Make the local copy now? [y/N]'
        if ($answer -notmatch '^[Yy]') {
            Say 'Nothing was copied. Re-run when you have time, or pass -Yes.' 'Yellow'
            exit 0
        }
    }
    $null = New-Item -ItemType Directory -Path $Local -Force
    Say 'Copying... (this is the only slow step, and only happens once)'
    # /MIR is safe here: $Local is created and owned by this script.
    $null = robocopy $Share $Local /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /NP /R:2 /W:5
    if ($LASTEXITCODE -ge 8) {
        Say "Copy failed (robocopy exit $LASTEXITCODE). Nothing else was done." 'Red'
        exit 1
    }
    Say 'Local copy ready.' 'Green'
    Write-Host ''
}

$sync = Join-Path $here 'Sync_My_Work.ps1'
if (-not (Test-Path -LiteralPath $sync)) { $sync = Join-Path $Local 'Sync_My_Work.ps1' }

# ------------------------------------------------------------- 1+2. sync down ---
Rule
Say 'SYNC  before painting' 'Cyan'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $sync -Share $Share -Local $Local
Rule
Write-Host ''

if ($SyncOnly) { Say 'Sync only - painter not started.' 'Yellow'; exit 0 }

# ------------------------------------------------------------------ 3. paint ----
$py = $null
foreach ($c in @((Get-Command python -ErrorAction SilentlyContinue | Where-Object { $_.Source -notmatch 'WindowsApps' } | Select-Object -First 1).Source,
                 'C:\Program Files\Python\python.exe')) {
    if ($c -and (Test-Path -LiteralPath $c)) { $py = $c; break }
}
if (-not $py) {
    Say 'Python 3.10+ was not found. Install it from python.org (tick "Add to PATH"),' 'Red'
    Say 'then run:  pip install numpy pillow' 'Red'
    exit 1
}
& $py -c 'import numpy, PIL' 2>$null
if ($LASTEXITCODE -ne 0) {
    Say "Python is at $py but numpy/pillow are missing. Run:  pip install numpy pillow" 'Red'
    exit 1
}

# Open on the first acquisition present; the painter's own dropdown reaches all the others.
$firstAcq = Get-ChildItem -LiteralPath (Join-Path $Local 'data_fullres') -Directory |
            Where-Object { $_.Name -like 'context4d_*' } | Select-Object -First 1
if (-not $firstAcq) { Say 'No acquisitions in the local copy - nothing to paint.' 'Red'; exit 1 }

Say 'PAINT' 'Cyan'
Say '  Open the address printed below on the iPad (same Wi-Fi).'
Say '  Press Ctrl-C in this window when you are finished -- your masks are' 'Yellow'
Say '  pushed back to the share automatically as soon as you do.' 'Yellow'
Rule
$env:PYTHONPATH = Join-Path $Local 'software\src'
try {
    & $py -m cc_reservoir.viz.ipad_paint $firstAcq.FullName --port $Port
} finally {
    # Runs on Ctrl-C too. A hard window close skips it, which is why the next run
    # syncs up before anything else.
    Write-Host ''
    Rule
    Say 'SYNC  after painting' 'Cyan'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $sync -Share $Share -Local $Local -SkipCode
    Rule
    Write-Host ''
    Say 'Your masks are on the share, in data_fullres.' 'Green'
    Say 'They reach 1_masks_USE_THESE when the merge is re-run.' 'Gray'
    Write-Host ''
}
