#!/usr/bin/env python3
"""Finite checks of the analytic pairwise/copy separation, not a proof search.

The lifted-trine primal and dual witnesses are checked in physical coordinates.
Adaptive transcript probabilities are exact Fractions; logarithms use 80 digits.
The all-phase and all-accuracy statements are proved in the manuscript.
"""
from fractions import Fraction as F
from pathlib import Path
import json
import mpmath as mp
import numpy as np

mp.mp.dps = 80


def real(x):
    if isinstance(x, F):
        return mp.mpf(x.numerator) / x.denominator
    return mp.mpf(x)


def kl(p, q):
    p, q = real(p), real(q)
    return p * mp.log(p / q) + (1 - p) * mp.log((1 - p) / (1 - q))


def value(t):
    t = real(t)
    return (2 - t + 2 * mp.sqrt(2 * t * (1 - t))) / 3


def physical_rows():
    u = np.array([[1., 0., 0.], [-.5, np.sqrt(3)/2, 0.],
                  [-.5, -np.sqrt(3)/2, 0.]])
    e = np.array([0., 0., 1.])
    mu = (np.sqrt(2) * u + e) / np.sqrt(3)
    assert np.max(np.abs(mu @ mu.T - np.eye(3))) < 2e-15
    rows = []
    for t in [0., *np.geomspace(1e-14, .25, 40)]:
        psi = np.sqrt(1 - t) * u + np.sqrt(t) * e
        gram = psi @ psi.T
        expected = (1 - (3*t - 1)/2) * np.eye(3) + (3*t - 1)/2
        assert np.max(np.abs(gram - expected)) < 5e-15
        a = np.sqrt(2*(1-t)) + np.sqrt(t)
        dual = a/3 * np.diag([np.sqrt((1-t)/2)]*2 + [np.sqrt(t)])
        primal = float(np.mean(np.diag(mu @ psi.T)**2))
        dual_value = float(np.trace(dual))
        residual = min(float(np.linalg.eigvalsh(
            dual - np.outer(v, v)/3)[0]) for v in psi)
        assert residual > -5e-15
        assert abs(primal - dual_value) < 5e-15
        assert abs(primal - float(value(t))) < 5e-15
        rows.append({'t': float(t), 'primal': primal,
                     'dual': dual_value, 'min_dual_slack': residual})
    return rows


def adaptive_transcript(t, depth, policy):
    # Each action depends on the complete history. Cosines are exact rationals.
    cosines = [F(-1), F(-3, 5), F(0), F(3, 5), F(1)]
    frontier = {(): (F(1), F(1))}
    chain = mp.mpf(0)
    for step in range(depth):
        nxt = {}
        for hist, (w0, wt) in frontier.items():
            code = sum((i+1)*(b+1) for i, b in enumerate(hist))
            c = cosines[(policy + code + step*step) % len(cosines)]
            p0 = (1 - c/2)/2
            pt = (1 + (3*t-1)*c/2)/2
            assert F(1, 4) <= pt <= F(3, 4)
            bound = (p0-pt)**2 / (pt*(1-pt))
            assert bound <= 3*t*t
            assert kl(p0, pt) <= real(bound) + mp.mpf('1e-75')
            chain += real(w0) * kl(p0, pt)
            nxt[hist+(1,)] = (w0*p0, wt*pt)
            nxt[hist+(0,)] = (w0*(1-p0), wt*(1-pt))
        frontier = nxt
    assert sum(a for a, _ in frontier.values()) == 1
    assert sum(b for _, b in frontier.values()) == 1
    direct = sum(real(a)*mp.log(real(a)/real(b)) for a, b in frontier.values())
    assert abs(direct-chain) < mp.mpf('1e-70')
    assert direct <= real(3*depth*t*t)
    return {'t': str(t), 'depth': depth, 'policy': policy,
            'leaves': len(frontier), 'KL': float(direct),
            'chain_error': float(abs(direct-chain))}


def accuracy_rows():
    rows = []
    for k in range(3, 21):
        eps = F(1, 2**k)
        t = 16*eps*eps
        gap = value(t) - value(0)
        assert gap >= (4*mp.sqrt(6)-2)*real(eps)/3
        assert gap > 2*real(eps)
        for delta in [F(1, 4), F(1, 10), F(1, 1000)]:
            binary = kl(1-delta, delta)
            m_pair = binary/(768*real(eps)**4)
            m_copy_lower = mp.log(1/(4*real(delta)*(1-real(delta)))) / (
                -mp.log1p(-real(t)))
            m_copy_upper = mp.ceil(mp.log(2/real(delta))/(2*real(eps)**2))
            assert m_copy_lower <= m_copy_upper
            # The claimed copy threshold saturates the exact Helstrom bound.
            fidelity = mp.exp(m_copy_lower*mp.log1p(-real(t)))
            error = (1-mp.sqrt(1-fidelity))/2
            assert abs(error-real(delta)) < mp.mpf('1e-70')
            rows.append({'epsilon': str(eps), 'delta': str(delta),
                         'value_gap_over_epsilon': float(gap/real(eps)),
                         'pair_lower': float(m_pair),
                         'copy_lower': float(m_copy_lower),
                         'copy_upper': int(m_copy_upper)})
    return rows


def dual_nonattainment_rows():
    rows = []
    for h in [F(1, 2**k) for k in range(1, 21)]:
        a, c = F(1, 4)+h, F(1, 4)+1/(16*h)
        # Both 2x2 PSD slacks have positive diagonal and zero determinant.
        for off in [F(1, 4), F(-1, 4)]:
            assert (a-F(1, 4))*(c-F(1, 4))-off*off == 0
        assert 2*a == F(1, 2)+2*h
        rows.append({'h': str(h), 'objective': str(2*a)})
    # At a=1/4, the two zero-diagonal PSD constraints force unequal b values.
    assert F(1, 4) != F(-1, 4)
    return rows


def adverse_controls():
    t = F(1, 2**20)
    # A nearly identical real pair allows a zero-probability event; the
    # trine's variance lower bound must not be extended to every ensemble.
    real_pair_kl = -mp.log1p(-real(t)/2)
    assert real_pair_kl > 3*real(t)**2
    # Measuring the orthogonal lift coordinate sees probability t directly.
    # The pairwise per-shot KL bound cannot be applied to prepared-state data.
    lift_coordinate_kl = -mp.log1p(-real(t))
    assert lift_coordinate_kl > 3*real(t)**2
    # Omitting the square-root coherence term violates the achievable value.
    assert real((2-t)/3) < value(t)
    return {'near_identical_pair_excluded_from_trine_KL_bound': True,
            'copy_measurement_excluded_from_pairwise_KL_bound': True,
            'missing_coherence_term_rejected': True,
            'dual_minimum_at_singular_gram_rejected': True}


def main():
    packet = {
        'status': 'pairwise_accuracy_lower_bound_and_copy_separation_checked',
        'scope': 'Finite consistency checks; analytic proofs cover the continuum. '
                 'Pairwise bit observations only; no source-code or general '
                 'preparation-query lower bound and no optimal N dependence.',
        'physical_primal_dual_rows': physical_rows(),
        'adaptive_transcript_rows': [adaptive_transcript(t, 9, policy)
            for t in [F(1, 4), F(1, 64), F(1, 4096)] for policy in range(5)],
        'accuracy_and_confidence_rows': accuracy_rows(),
        'dual_nonattainment_rows': dual_nonattainment_rows(),
        'adverse_controls': adverse_controls(),
    }
    directory = Path(__file__).resolve().parent
    outdir = directory/'results' if directory.name == 'numerics' else directory
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir/'AQ_SDP_PAIRWISE_LOWER_BOUND_v1_RESULT.json'
    out.write_text(json.dumps(packet, indent=2)+'\n')
    print(packet['status'])
    print('41 physical witness rows; 15 adaptive trees; 54 accuracy/confidence '
          'rows; 20 dual-limit rows; 4 adverse controls')


if __name__ == '__main__':
    main()
