import json, sys

for path in ["notebooks/03_search_api_benchmark.ipynb", "notebooks/04_feast_feature_store.ipynb"]:
    print(f"\n=== {path} ===")
    nb = json.load(open(path, encoding="utf-8"))
    found = False
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            otype = out.get("output_type")
            if otype == "error":
                found = True
                print(f"  Cell {i} EXCEPTION [{cell.get('source','')[:50]}...]:")
                print(f"    {out.get('ename')}: {out.get('evalue','')[:200]}")
                tb = out.get("traceback", [])
                for line in tb[-6:]:
                    clean = ""
                    esc = False
                    for ch in line:
                        if ch == "\x1b": esc = True
                        elif esc and ch == "m": esc = False
                        elif not esc: clean += ch
                    if clean.strip():
                        print("    |", clean.strip()[:120])
    if not found:
        print("  (no error cells found in stored outputs)")
