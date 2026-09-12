# Quantum Gram Stability Certificates

Audit programs accompanying Zhongpai Gao's paper *Finite-Sample Stability for Gram-Reduced Bayes-Optimal Quantum State Discrimination*.
This repository is self-contained; no research-workspace checkout is required.

## Programs

| Program | What it checks |
| --- | --- |
| `state_discrimination_bayes_gram_stability_audit.py` | 99 finite checks of Bayes-value continuity, Gram repair, sampling budgets and primal/dual intervals. |
| `state_discrimination_pairwise_lower_bound_audit.py` | Lifted-trine witnesses, adaptive transcripts, pairwise sampling bounds and observation-model controls. |
| `state_discrimination_coherent_gram_audit.py` | Diagonal drift, complex index preparation, fidelity conversions and copy budgets. |
| `state_discrimination_direct_value_audit.py` | Flat-basis witnesses, multinomial laws, error budgets and adverse promise controls. |

These are finite numerical and exact-arithmetic consistency checks, not formal eigenvalue certificates or a proof of the continuum theorems. The programs do not implement the inherited collective tomography measurement and do not certify quantum advantage or unrestricted sample optimality.

The 8 September working revision also checks gap-sensitive Bayes stability: 48 matrix cases, 36 sampling-budget cases, and a false-gap adverse control. A supplied full-Gram floor `G >= kappa I` yields `O(N^3 kappa^-1 epsilon^-2 log(N/delta))` pairwise shots. A certified positive floor of the computed repair sharpens its posterior interval without assuming a true-Gram gap. These refinements do not replace the gap-free theorem.

## Run

Python 3.12 or later for the pinned NumPy version, plus mpmath. The reference environment is Python 3.13.2, NumPy 2.5.1 and mpmath 1.4.1. Install the reference package versions with:

```sh
python3 -m pip install -r requirements.txt
```

From this directory, run all audits:

```sh
python3 run_audits.py
```

Each audit can also be run separately with `python3 SCRIPT_NAME.py`.
A failed assertion stops the run with a nonzero exit status. Successful completion
prints `All 4 audits passed.` The programs regenerate their corresponding
JSON result files beside the scripts, overwriting the supplied reference copies.
Keep this flat layout; do not add a `results/` directory.

## Reference results and integrity

The JSON outputs are included. `PROVENANCE.json` records their original research
workspace paths, hashes and reference environment; those paths are provenance
only and are not needed to run the code. No research-repository checkout is required.

Before running, verify the supplied snapshot on Linux with:

```sh
sha256sum -c SHA256SUMS
```

The audit instances are deterministic. Floating-point result bytes can vary across NumPy, BLAS, operating systems and processors; the assertions use the tolerances specified in each program. A checksum mismatch after regeneration alone does not establish a mathematical failure. The supplied checksums identify the original snapshot.

## GitHub release and citation

Repository: [quantum-gram-stability-certificates](https://github.com/Gaozhongpai/quantum-gram-stability-certificates).

`CITATION.cff` supplies citation metadata. Cite the accompanying paper by title
and identify the software commit or versioned release used. The reference
snapshot is dated 8 September 2026 (local working revision).

## License

The audit programs and documentation use the [MIT License](LICENSE), matching
the author's spin-squarefactor certificate repository.
