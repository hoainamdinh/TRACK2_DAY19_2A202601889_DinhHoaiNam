"""Re-execute NB03 and NB04 via nbconvert and capture errors."""
import subprocess, sys
from pathlib import Path

ROOT = Path(".").resolve()
venv = ROOT / ".venv"
py = venv / "bin" / "python"
nbconvert = venv / "bin" / "jupyter"

for nb_py, nb_ipynb in [
    ("notebooks/03_search_api_benchmark.py", "notebooks/03_search_api_benchmark.ipynb"),
    ("notebooks/04_feast_feature_store.py", "notebooks/04_feast_feature_store.ipynb"),
]:
    print(f"\n=== Converting {nb_py} ===")
    # Convert .py -> .ipynb
    r = subprocess.run(
        [str(venv/"bin"/"jupytext"), "--to", "notebook", "--update", nb_py],
        cwd=str(ROOT), capture_output=True, text=True
    )
    print(r.stdout[:200], r.stderr[:200])

    print(f"=== Executing {nb_ipynb} ===")
    r = subprocess.run(
        [str(nbconvert), "nbconvert", "--to", "notebook", "--execute",
         "--inplace", nb_ipynb, "--ExecutePreprocessor.timeout=300"],
        cwd=str(ROOT), capture_output=True, text=True
    )
    print("returncode:", r.returncode)
    print(r.stdout[-500:])
    print(r.stderr[-500:])
