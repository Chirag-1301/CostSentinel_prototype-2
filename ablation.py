"""Optional ablation-only run. Use run_all.py to rebuild the complete submission."""
import costsentinel_prototype as cs
from threadpoolctl import threadpool_limits

if __name__ == "__main__":
    with threadpool_limits(limits=1):
        cs.run_ablations()
    print("Ablation CSV regenerated. Run run_all.py for synchronized dashboard data.")
