"""End-to-end /save round trip -- the thing the selftest cannot reach.

The selftest only calls seed_stamp() directly, which is why a np.int64 in the /save response
sailed through it and turned every Save into an HTTP 500. This drives the real Handler over
a real socket and asserts the status code and the body.
"""
import json
import os
import sys
import tempfile
import threading
import urllib.request

import numpy as np

sys.path.insert(0, r"C:\Storage\CisternaChyli\cisterna-chyli-reservoir\src")
from cc_reservoir.viz import ipad_paint as ip

TMP = os.path.join(tempfile.gettempdir(), "cc_http_save_test")
ACQ = os.path.join(TMP, "context4d_07_20_22_data_TestAcq")
os.makedirs(ACQ, exist_ok=True)
# start clean: a leftover sidecar makes the Session resume from it, so the "unedited"
# first save would differ from the seed and the assertion below would misfire
import glob as _g
for _f in _g.glob(os.path.join(ACQ, "*_edit.npz")):
    os.remove(_f)

shape = (40, 50, 60)
rng = np.random.default_rng(0)
for f in ("f0", "f1"):
    hu = (rng.normal(-100, 30, shape)).astype(np.int16)
    hu[20, 25, 28:34] = 600                       # a bright "duct" to paint on
    seed = np.zeros(shape, np.uint8)
    seed[20, 25, 30] = 1                          # a 1-voxel auto seed
    np.savez_compressed(os.path.join(ACQ, f + ".npz"), hu=hu, seg=seed,
                        voxel=np.array([0.8, 0.8, 0.5]))
np.savez_compressed(os.path.join(ACQ, "meta.npz"), voxel=np.array([0.8, 0.8, 0.5]),
                    crop_lo=np.zeros(3, np.int32), crop_hi=np.array(shape, np.int32),
                    raw_full_shape=np.array(shape, np.int32), n_frames=np.int64(2))

PORT = 8791
srv = ip.ThreadingHTTPServer(("127.0.0.1", PORT), ip.Handler)
srv.sess = ip.Session(ACQ)          # same wiring as ipad_paint.serve()
threading.Thread(target=srv.serve_forever, daemon=True).start()


def post(path, payload=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload or {}).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


fail = []
code, body = post("/save")
print("POST /save (unedited frame) ->", code, body if isinstance(body, str) else
      {k: body[k] for k in ("ok", "voxels", "diff_seed") if k in body})
if code != 200:
    fail.append(f"unedited save returned {code}: {body}")
elif body.get("diff_seed") != 0:
    fail.append(f"unedited save should report diff_seed 0, got {body.get('diff_seed')}")

# Paint by mutating the session's seg directly: this test is about the /save contract, and
# guessing display coordinates for /stroke only tests my arithmetic.
with srv.sess.lock:
    srv.sess.seg[20, 25, 31] = 2
    srv.sess._count[2] = int((srv.sess.seg == 2).sum())
print("painted 1 voxel directly into the session seg")

code3, body3 = post("/save")
print("POST /save (after a stroke) ->", code3, body3 if isinstance(body3, str) else
      {k: body3[k] for k in ("ok", "voxels", "diff_seed") if k in body3})
if code3 != 200:
    fail.append(f"edited save returned {code3}: {body3}")
elif not isinstance(body3.get("diff_seed"), int) or body3["diff_seed"] <= 0:
    fail.append(f"edited save should report a positive int diff_seed, got {body3.get('diff_seed')!r}")

side = os.path.join(ACQ, "f0_edit.npz")
if os.path.exists(side):
    with np.load(side, allow_pickle=True) as d:
        print("sidecar keys:", list(d.files),
              "edited=", bool(d["edited"]), "n_diff_seed=", int(d["n_diff_seed"]))

srv.shutdown()
print()
if fail:
    for f in fail:
        print("FAIL:", f)
    raise SystemExit(1)
print("HTTP /save round trip OK")
