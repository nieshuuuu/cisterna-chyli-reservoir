# sync-cc.ps1
# Keep the local cisterna-chyli-reservoir clone current, refresh cc_paint's embedded
# package from it, and push that package to the painter bundle on the share so
# collaborators pick up code fixes without anyone copying files by hand.
# Safe to run repeatedly; does nothing costly when the repo is already up to date.
# Registered as a Windows Scheduled Task.
#
# Design notes:
#  - Pull is --ff-only: it never rewrites or discards anything. If the clone ever
#    diverges (someone commits locally), the pull fails and we log it rather than
#    clobbering work.
#  - The embedded package (cc_paint\src\cc_reservoir) is a mirror of the repo and
#    is safe to overwrite. It is kept because cc_reservoir.scripts.backfill_provenance
#    documents that path on PYTHONPATH for the masks in cc_paint\data.
#  - The TOP-LEVEL runner (cc_paint\ipad_paint.py) was RETIRED on 2026-09-03: it was a
#    standalone copy frozen at 3db66e8 (2026-07-17), verified byte-identical to that
#    commit, so the "may hold local edits" caution this script logged for six weeks was
#    never true of it. Archived under C:\Storage\ClaudeData\. The check below is kept
#    for the case where someone drops a runner back in; it is silent when there is none.
#  - The share is addressed by UNC, NOT by the Z: drive letter: a scheduled task
#    does not necessarily have the user's mapped drives. Z: is only a fallback.
#  - Only `software\src\cc_reservoir` in the bundle is mirrored. The bundle's
#    data_fullres (masks + CT) is never touched by this script.

$git    = if (Test-Path 'C:\Program Files\Git\cmd\git.exe') { 'C:\Program Files\Git\cmd\git.exe' } else { 'git' }
$base   = 'C:\Storage\CisternaChyli'
$repo   = Join-Path $base 'cisterna-chyli-reservoir'
# cc_paint could not be moved with the rest (2026-07-20: a stale painter on port 8000
# holds a lock on it). Auto-detect: once it lands in $base this picks it up unchanged.
$paint  = if (Test-Path (Join-Path $base 'cc_paint')) { Join-Path $base 'cc_paint' } else { 'C:\Users\nies1\cc_paint' }
$pkgSrc = Join-Path $repo 'src\cc_reservoir'
$pkgDst = Join-Path $paint 'src\cc_reservoir'
$topRun = Join-Path $paint 'ipad_paint.py'
$log    = Join-Path $base 'cc_autosync\sync.log'

$env:GIT_TERMINAL_PROMPT = '0'   # never block on a credential prompt

function Log($m) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Add-Content -Path $log -Value "$ts  $m"
}

if (-not (Test-Path $repo)) { Log "ERROR: clone missing at $repo"; exit 1 }

# --- pull (stderr discarded to avoid PS 5.1 native-error wrapping; status via exit code + HEAD) ---
$before = (& $git -C $repo rev-parse HEAD 2>$null)
$pullOut = (& $git -C $repo pull --ff-only 2>$null)
$pullCode = $LASTEXITCODE
$after  = (& $git -C $repo rev-parse HEAD 2>$null)

if ($pullCode -ne 0) {
    Log "WARN: 'git pull --ff-only' exit $pullCode (offline, auth, or diverged) - keeping package consistent with current HEAD $($after.Substring(0,8))"
} elseif ($before -ne $after) {
    $subj = (& $git -C $repo log -1 --format='%s' 2>$null)
    Log "updated: $($before.Substring(0,8)) -> $($after.Substring(0,8))  ($subj)"
} else {
    Log "already current at $($after.Substring(0,8))"
}

# --- mirror the package into cc_paint (idempotent; near-instant when unchanged) ---
$null = robocopy $pkgSrc $pkgDst /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /NP
$rc = $LASTEXITCODE
if ($rc -ge 8) { Log "ERROR: robocopy exit $rc mirroring package into cc_paint" }
elseif ($rc -band 1) { Log "cc_paint embedded package refreshed (files copied)" }

# --- push the package to the painter bundle on the share ---------------------
# Without this the bundle stays frozen at whatever was packaged by hand, so anyone
# launching the painter from the share (or refreshing a local copy from it) runs
# stale code. Mirroring is safe: this folder is a copy of the repo package, nothing
# is authored there.
$shareRoots = @(
    '\\160.87.12.113\Molloilab\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation\2_CT_volumes_and_painter',
    'Z:\ImageData\Lymph_Studies\Cisterna_Chyli_and_Thoracic_Duct_Segmentation\2_CT_volumes_and_painter'
)
$bundle = $null
foreach ($r in $shareRoots) {
    if (Test-Path $r) { $bundle = $r; break }
}
if (-not $bundle) {
    Log "NOTE: share not reachable - painter bundle not refreshed this run"
} else {
    # Publish the COMMITTED tree, never the working tree.
    #
    # This clone is a development checkout and is dirty most of the time. Copying
    # $repo\src straight to the share pushed half-finished edits and untracked scratch
    # files out to collaborators while the log reported the bundle as consistent with
    # HEAD. `git archive` exports exactly what is committed. Skipping the refresh while
    # the tree is dirty was the alternative and is worse: the tree is nearly always
    # dirty, so the bundle would freeze and collaborators would quietly stop receiving
    # real fixes -- a silent no-content bug traded for a silent wrong-content one.
    $stage = Join-Path $env:TEMP ('cc_pkg_' + $after.Substring(0, 8))
    $bundleTools = $null
    try {
        if (Test-Path $stage) { Remove-Item -Recurse -Force -Path $stage }
        $null = New-Item -ItemType Directory -Path $stage -Force
        $tarPath = Join-Path $stage 'pkg.tar'
        # Build the pathspec from what actually exists at HEAD. `git archive` fails
        # outright on an unknown path, and one not-yet-committed folder must not be able
        # to block the package refresh for everyone.
        $arcPaths = @()
        foreach ($p in @('src/cc_reservoir', 'tools/painter_bundle', 'tools/share_docs')) {
            & $git -C $repo cat-file -e "HEAD:$p" 2>$null
            if ($LASTEXITCODE -eq 0) { $arcPaths += $p } else { Log "NOTE: $p not in HEAD - not published this run" }
        }
        if ($arcPaths -notcontains 'src/cc_reservoir') {
            Log "ERROR: HEAD has no src/cc_reservoir - share bundle NOT refreshed"
            $arc = 1
        } else {
            & $git -C $repo archive --format=tar -o $tarPath HEAD @arcPaths 2>$null
            $arc = $LASTEXITCODE
        }
        if ($arc -ne 0) {
            Log "ERROR: git archive exit $arc - share bundle NOT refreshed (nothing published)"
        } else {
            & tar -x -f $tarPath -C $stage
            # git archive keeps the full repo-relative prefix, so the export lands at
            # <stage>\src\cc_reservoir and <stage>\tools\painter_bundle. Mirroring <stage>
            # itself would add a nesting level and, because of /MIR, delete every real
            # module from the bundle on the way past.
            $cleanPkg = Join-Path $stage 'src\cc_reservoir'
            $bundleTools = Join-Path $stage 'tools\painter_bundle'
            if (-not (Test-Path $cleanPkg)) {
                Log "ERROR: export is missing src\cc_reservoir - share bundle NOT refreshed"
            } else {
                $bundleDst = Join-Path $bundle 'software\src\cc_reservoir'
                $null = robocopy $cleanPkg $bundleDst /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /NP /R:2 /W:2
                $rcb = $LASTEXITCODE
                if ($rcb -ge 8) {
                    Log "ERROR: robocopy exit $rcb pushing package to the share bundle"
                } elseif ($rcb -band 1) {
                    Log "share painter bundle refreshed at $($after.Substring(0,8)) -> $bundleDst"
                }
            }
        }
    } catch {
        Log "ERROR: staging the clean export failed - $($_.Exception.Message)"
        $bundleTools = $null
    }

    # --- deploy the operator-facing files into folders that hold real data -----
    # Copied file by file, and NEVER with /MIR. Every destination below sits beside
    # something irreplaceable -- the bundle root holds data_fullres (16 GB of CT and
    # masks) and software\, and the segmentation root holds all three mask folders --
    # so a mirror would delete them.
    #
    # Compared by CONTENT HASH, not by timestamp. robocopy's default rule skips a
    # destination that is newer than the source, so an export whose mtimes land
    # after the copy on the share would silently deploy nothing - which is exactly
    # what happened the first time this was written.
    function Deploy-Files($srcDir, $dstDir, $label) {
        if (-not $srcDir -or -not (Test-Path $srcDir)) { return }
        if (-not (Test-Path $dstDir)) {
            Log "NOTE: $label destination missing ($dstDir) - skipped"
            return
        }
        $done = @()
        foreach ($f in Get-ChildItem -Path $srcDir -File) {
            $dstFile = Join-Path $dstDir $f.Name
            $need = $true
            if (Test-Path $dstFile) {
                $need = (Get-FileHash $f.FullName -Algorithm SHA256).Hash -ne
                        (Get-FileHash $dstFile -Algorithm SHA256).Hash
            }
            if ($need) {
                try {
                    Copy-Item -Path $f.FullName -Destination $dstFile -Force -ErrorAction Stop
                    $done += $f.Name
                } catch {
                    Log "ERROR: could not deploy $($f.Name) to $label - $($_.Exception.Message)"
                }
            }
        }
        if ($done.Count -gt 0) { Log "$label deployed ($($done -join ', '))" }
    }

    Deploy-Files $bundleTools $bundle 'bundle launchers/sync scripts'

    # --- deploy the share's own documentation ----------------------------------
    # tools\share_docs\<folder>\ maps to <segmentation root>\<folder>\, with the
    # special name "root" meaning the segmentation root itself. These READMEs are what
    # a new person reads first, and they were drifting on the share with no history.
    # Only files that already have a destination folder are copied, so a renamed
    # folder is reported rather than silently recreated in the wrong place.
    $shareDocs = Join-Path $stage 'tools\share_docs'
    if ((Test-Path $shareDocs) -and $bundle) {
        $segRoot = Split-Path $bundle -Parent
        foreach ($d in Get-ChildItem -Path $shareDocs -Directory) {
            $dst = if ($d.Name -eq 'root') { $segRoot } else { Join-Path $segRoot $d.Name }
            Deploy-Files $d.FullName $dst "share docs -> $($d.Name)"
        }
    }

    # The clean export is only needed until both copies above are done. Keyed on the
    # HEAD sha, so a crashed run leaves at most one stale directory per commit.
    if (Test-Path $stage) { Remove-Item -Recurse -Force -Path $stage -ErrorAction SilentlyContinue }
}

# --- report drift between this running script and its versioned copy ---------
# The scheduled task runs the deployed copy, not the repo one: a script that
# overwrites itself mid-run is a bad idea, and this copy holds machine-specific
# paths. Same rule as cc_paint\ipad_paint.py - report, never overwrite.
$selfRepo = Join-Path $repo 'tools\sync-cc.ps1'
if ((Test-Path $selfRepo) -and $PSCommandPath) {
    $hRun = (Get-FileHash $PSCommandPath -Algorithm SHA256).Hash
    $hRepo = (Get-FileHash $selfRepo -Algorithm SHA256).Hash
    if ($hRun -ne $hRepo) {
        Log "NOTE: cc_autosync\sync-cc.ps1 differs from tools\sync-cc.ps1 in the repo - copy it over when ready."
    }
}

# --- report drift on a top-level runner, if anyone puts one back; never overwrite it ---
# Retired 2026-09-03 (see the header). Normally this block does nothing at all: painting goes
# through the share's Paint.bat, which runs software\src, not a private copy. If a runner
# reappears here it is almost certainly someone about to paint with stale code, so say so.
if (Test-Path $topRun) {
    $repoBlob = (& $git -C $repo hash-object 'src/cc_reservoir/viz/ipad_paint.py' 2>$null)
    $topBlob = (& $git hash-object $topRun 2>$null)
    if ($topBlob -ne $repoBlob) {
        Log "NOTE: cc_paint\ipad_paint.py is back and differs from the repo painter - left untouched. Paint via the share's Paint.bat, not this file."
    }
}
