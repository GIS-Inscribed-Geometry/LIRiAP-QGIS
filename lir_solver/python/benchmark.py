"""Quick benchmark + gap measurement. Usage: python benchmark.py [test.geojson]"""
import json, sys, time, pathlib
from shapely.geometry import shape, box
from shapely.prepared import prep

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

PACK = HERE.parent.parent / "LIRiAP-QGIS" / "LIRiAP_pack"
if PACK.exists(): sys.path.insert(0, str(PACK))

from lir_rust_shim import solve_lir_shim, _RUST_AVAILABLE

GJ = pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parent.parent/"testt_cases-2.geojson"
PARAMS = dict(axis_angle=0, grid_fine=300, max_ratio=1.6,
              always_return=True, use_buffer=True, buf_value=-0.5)

def gaps(poly, rect):
    if rect is None: return {}
    pp = prep(poly); bx0,by0,bx1,by1 = rect.bounds; g = {}
    for s in "LRBT":
        lo,hi = 0.0,30.0
        for _ in range(50):
            m=0.5*(lo+hi)
            c = box(bx0-m,by0,bx1,by1) if s=="L" else \
                box(bx0,by0,bx1+m,by1) if s=="R" else \
                box(bx0,by0-m,bx1,by1) if s=="B" else box(bx0,by0,bx1,by1+m)
            if pp.covers(c): lo=m
            else: hi=m
        g[s]=lo
    return g

with open(GJ) as f: feats = json.load(f)["features"]
print(f"{len(feats)} features | Rust available: {_RUST_AVAILABLE}")

t0 = time.perf_counter()
results = [solve_lir_shim(shape(ft["geometry"]), **PARAMS) for ft in feats]
print(f"Rust: {time.perf_counter()-t0:.3f}s")

print(f"\n{'fid':>5}  {'type':20}  {'area':>8}  {'ratio':>6}  L     R     B     T")
for ft, (rect,area,_,pt,ratio,be) in zip(feats[:10], results[:10]):
    poly = shape(ft["geometry"])
    g = gaps(poly, rect)
    fid = ft["properties"].get("fid","?")
    if rect is None: print(f"{fid:>5}  None"); continue
    print(f"{fid:>5}  {pt:20}  {area:8.0f}  {ratio:6.3f}  "
          f"{g.get('L',0):.2f}  {g.get('R',0):.2f}  {g.get('B',0):.2f}  {g.get('T',0):.2f}")
