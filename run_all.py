"""Reproduce all simulation evidence, then rebuild the offline dashboard."""
import os
import subprocess
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

if __name__ == "__main__":
    import costsentinel_prototype as cs
    cs.main()
    subprocess.run([sys.executable, str(Path(__file__).with_name("build_dashboard.py"))], check=True)
    print("Complete: CSV/JSON evidence and HTML dashboard rebuilt together.")
