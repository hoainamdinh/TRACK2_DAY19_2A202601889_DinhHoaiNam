import json, sys

for path in ["notebooks/03_search_api_benchmark.ipynb", "notebooks/04_feast_feature_store.ipynb"]:
    print(f"\n=== {path} ===")
    try:
        nb = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        print(f"  CANNOT READ: {e}")
        continue
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            if otype == "error":
                print(f"  Cell {i} EXCEPTION: {out.get('ename')}: {out.get('evalue','')[:120]}")
                for line in out.get("traceback", [])[-4:]:
                    clean = ""
                    skip = False
                    for ch in line:
                        if ch == "\x1b":
                            skip = True
                        elif skip and ch == "m":
                            skip = False
                        elif not skip:
                            clean += ch
                    print("    " + clean[:120])
            elif otype == "stream":
                text = "".join(out.get("text", []))
                if any(k in text for k in ("WARN", "Error", "FAIL", "error", "Traceback", "assert")):
                    print(f"  Cell {i} STREAM: {text[:400]}")
