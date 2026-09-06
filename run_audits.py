#!/usr/bin/env python3
"""Run the supplied audit programs; stop immediately on a failed audit."""
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    scripts = ['state_discrimination_bayes_gram_stability_audit.py', 'state_discrimination_pairwise_lower_bound_audit.py', 'state_discrimination_coherent_gram_audit.py', 'state_discrimination_direct_value_audit.py']
    for script in scripts:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(here / script)], cwd=here, check=True)
    print(f"All {len(scripts)} audits passed.")


if __name__ == "__main__":
    main()
