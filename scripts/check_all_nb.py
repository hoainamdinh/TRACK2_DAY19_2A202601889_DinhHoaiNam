import json, glob, sys

errors = []
passes = []
for path in sorted(glob.glob("notebooks/*.ipynb")):
    nb = json.load(open(path, encoding="utf-8"))
    has_error = False
    has_output = False
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            has_output = True
            if out.get("output_type") == "error":
                has_error = True
                errors.append(f"FAIL {path}: {out['ename']}: {out['evalue'][:80]}")
    if not has_output:
        errors.append(f"EMPTY {path}: no outputs (not executed?)")
    elif not has_error:
        passes.append(f"OK   {path}")

for p in passes:
    print(p)
for e in errors:
    print(e)
if errors:
    sys.exit(1)
