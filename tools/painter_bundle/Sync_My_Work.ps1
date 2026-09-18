# Sync_My_Work.ps1
# Reconciles painted masks between YOUR LOCAL COPY and the share, and refreshes the
# painter code on the way.
#
# You normally do not run this yourself -- Paint.bat calls it before and after every
# painting session. Run it directly only to sync without painting.
#
# HOW IT DECIDES, and why it is not just "newest wins":
#   It remembers, in .sync_state.json next to your local copy, the content hash each mask
#   had at the last successful sync. Comparing both sides against that record tells the
#   three cases apart:
#       only YOUR side changed    -> push up
#       only the SHARE changed    -> pull down
#       BOTH changed              -> conflict: neither is touched, and it is reported
#   A newest-wins rule cannot see the difference between "I edited after pulling theirs"
#   and "we both edited the same frame", so it quietly destroys one of them. That is not
#   hypothetical -- the first version of this script did exactly that: it reported the
#   conflict on the way up and then overwrote the local mask on the way down.
#
# Nothing is ever deleted, on either side. Only f*_edit.npz files move; the CT volumes
# are never copied by this script.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\Sync_My_Work.ps1
#         ... -Preview     show what would happen, change nothing

param(
    [string] $Share,
    [string] $Local,
    # Deliberately NOT named -WhatIf: that is a PowerShell common parameter, and declaring
    # one by hand next to [CmdletBinding()] fights the engine's own handling of it.
    [switch] $Preview,
    [switch] $SkipCode
)

$ErrorActionPreference = 'Stop'
if (-not $Local) { $Local = $PSScriptRoot }

# Every path below goes through -LiteralPath. A local copy under a folder containing [ or ]
# -- "D:\pigs [2026]\painter" is entirely ordinary -- is otherwise parsed as a wildcard
# character class: Get-ChildItem matches nothing and the sync silently does nothing.

if (-not $Share) {
    # UNC first: a drive letter is per-user and may not be mapped.
    $candidates = @(
        '\\160.87.12.113\Molloilab\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation\2_CT_volumes_and_painter',
        'Z:\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation\2_CT_volumes_and_painter'
    )
    foreach ($c in $candidates) { if (Test-Path -LiteralPath $c) { $Share = $c; break } }
}
if (-not $Share -or -not (Test-Path -LiteralPath $Share)) {
    Write-Host 'ERROR: cannot reach the share. Connect to the lab network (or map Z:) and try again.' -ForegroundColor Red
    exit 1
}

# Compare the two roots by identity, not as strings: one may be a UNC path and the other a
# mapped drive letter pointing at the very same folder.
function Get-RootId([string] $p) {
    $full = (Get-Item -LiteralPath $p).FullName.TrimEnd('\')
    if ($full -match '^[A-Za-z]:') {
        $drive = Get-PSDrive -Name $full.Substring(0, 1) -ErrorAction SilentlyContinue
        if ($drive -and $drive.DisplayRoot) { $full = $drive.DisplayRoot.TrimEnd('\') + $full.Substring(2) }
    }
    return $full.ToLowerInvariant()
}
if ((Test-Path -LiteralPath $Local) -and (Get-RootId $Local) -eq (Get-RootId $Share)) {
    Write-Host 'Local and share are the same folder, so there is nothing to sync.' -ForegroundColor Yellow
    exit 0
}

function Get-Hash([string] $p) {
    if (-not (Test-Path -LiteralPath $p)) { return $null }
    return (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash
}

function Get-Masks([string] $root) {
    $out = @{}
    if (-not (Test-Path -LiteralPath $root)) { return $out }
    foreach ($f in @(Get-ChildItem -LiteralPath $root -Recurse -Filter 'f*_edit.npz' -File -ErrorAction SilentlyContinue)) {
        $out[$f.FullName.Substring($root.Length).TrimStart('\')] = $f.FullName
    }
    return $out
}

function Copy-Mask([string] $from, [string] $to) {
    $dir = Split-Path $to -Parent
    if (-not (Test-Path -LiteralPath $dir)) { $null = New-Item -ItemType Directory -Path $dir -Force }
    Copy-Item -LiteralPath $from -Destination $to -Force
}

# ------------------------------------------------------------------- code ------
if (-not $SkipCode) {
    $src = Join-Path $Share 'software\src\cc_reservoir'
    $dst = Join-Path $Local 'software\src\cc_reservoir'
    if (Test-Path -LiteralPath $src) {
        # /MIR is deliberate and confined: this subtree is a copy of the repo package and
        # nothing is authored in it, so making it match the share exactly is the point. It
        # cannot reach data_fullres, which lives outside this path. Local edits to the
        # painter package ARE discarded here -- edit the repo, not your copy.
        # $rcArgs, not $args: $args is an automatic variable in PowerShell.
        $rcArgs = @($src, $dst, '/MIR', '/XD', '__pycache__', '/NFL', '/NDL', '/NJH', '/NJS', '/NP', '/R:2', '/W:2')
        if ($Preview) { $rcArgs += '/L' }
        $null = & robocopy @rcArgs
        $rc = $LASTEXITCODE
        if ($rc -ge 8) { Write-Host "   CODE  robocopy exit $rc - painter NOT updated" -ForegroundColor Red }
        elseif ($rc -band 1) { Write-Host '   CODE  painter updated from the share' -ForegroundColor Green }
        else { Write-Host '   CODE  already current' -ForegroundColor Green }
    }
}

# ------------------------------------------------------------------ masks ------
$localData = Join-Path $Local 'data_fullres'
$shareData = Join-Path $Share 'data_fullres'
$statePath = Join-Path $Local '.sync_state.json'

$state = @{}
if (Test-Path -LiteralPath $statePath) {
    try {
        $raw = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        foreach ($p in $raw.PSObject.Properties) { $state[$p.Name] = $p.Value }
    } catch {
        Write-Host '   NOTE  .sync_state.json unreadable - treating this as a first sync' -ForegroundColor Yellow
    }
}

$lm = Get-Masks $localData
$sm = Get-Masks $shareData
$rels = @($lm.Keys) + @($sm.Keys) | Sort-Object -Unique

$up = 0; $down = 0; $same = 0; $conflicts = @()
foreach ($rel in $rels) {
    $lp = Join-Path $localData $rel
    $sp = Join-Path $shareData $rel
    $lh = Get-Hash $lp
    $sh = Get-Hash $sp
    $bh = $state[$rel]

    if ($lh -and $sh -and $lh -eq $sh) { $same++; $state[$rel] = $lh; continue }

    $localChanged = ($lh -ne $bh)
    $shareChanged = ($sh -ne $bh)

    if ($localChanged -and -not $shareChanged) {
        if (-not $Preview) { Copy-Mask $lp $sp }
        Write-Host ("      up    {0}" -f $rel)
        $state[$rel] = $lh; $up++
    } elseif ($shareChanged -and -not $localChanged) {
        if (-not $Preview) { Copy-Mask $sp $lp }
        Write-Host ("      down  {0}" -f $rel)
        $state[$rel] = $sh; $down++
    } else {
        # both sides moved away from the last synced state - do not pick a winner
        $conflicts += [pscustomobject]@{
            Frame = $rel
            Mine  = if ($lh) { (Get-Item -LiteralPath $lp).LastWriteTime } else { 'missing' }
            Share = if ($sh) { (Get-Item -LiteralPath $sp).LastWriteTime } else { 'missing' }
        }
    }
}

if (-not $Preview) {
    $null = New-Item -ItemType Directory -Path (Split-Path $statePath -Parent) -Force -ErrorAction SilentlyContinue
    ($state | ConvertTo-Json -Depth 3) | Set-Content -LiteralPath $statePath -Encoding UTF8
}

Write-Host ("   MASKS {0} up, {1} down, {2} already matching, {3} conflicts" -f
            $up, $down, $same, $conflicts.Count) -ForegroundColor Green
if ($conflicts.Count) {
    Write-Host ''
    Write-Host '   CONFLICT - these changed on BOTH sides since the last sync.' -ForegroundColor Yellow
    Write-Host '   Neither copy was touched. Decide by hand before painting them again:' -ForegroundColor Yellow
    $conflicts | Format-Table -AutoSize
}
