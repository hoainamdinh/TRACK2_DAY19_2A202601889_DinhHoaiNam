import json, sys
nb = json.load(open("notebooks/03_search_api_benchmark.ipynb", encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            print(f"Cell {i} ERROR: {out['ename']}: {out['evalue']}")
            print("  " + "  ".join(out.get("traceback", [])[-3:]))
        elif out.get("output_type") in ("stream", "execute_result"):
            text = "".join(out.get("text", out.get("data",{}).get("text/plain",[])))
            if text.strip():
                print(f"Cell {i}: {text[:300]}")
