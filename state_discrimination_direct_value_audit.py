#!/usr/bin/env python3
"""Finite checks for the direct flat-basis value theorem.

Dense physical primal/dual witnesses, exhaustive small multinomial laws,
and high-precision lower-bound budgets check different parts of the proof.
This is not an independent referee review or a numerical continuum proof.
"""
from pathlib import Path
from math import comb, factorial, isqrt, log
import json
import mpmath as mp
import numpy as np

mp.mp.dps = 80
RNG = np.random.default_rng(2026090601)


def fourier(n):
    return np.exp(2j*np.pi*np.outer(np.arange(n), np.arange(n))/n)/np.sqrt(n)


def value(p):
    return float(np.sqrt(np.maximum(p, 0)).sum()**2/len(p))


def root(a):
    w, v = np.linalg.eigh((a+a.conj().T)/2)
    assert min(w) > -1e-11
    return (v*np.sqrt(np.maximum(w, 0)))@v.conj().T


def bures(a, b):
    sa, sb = root(a), root(b)
    u, _, vh = np.linalg.svd(sa@sb)
    return float(np.linalg.norm(sa-u@vh@sb))


def witnesses():
    rows = []
    for n in [2, 3, 4, 5, 8]:
        for k in range(8):
            u = np.diag(np.exp(1j*RNG.normal(size=n)))@fourier(n)
            p = RNG.dirichlet(np.ones(n))
            if k < n:
                p[k+1:] = 0
                p /= p.sum()
            s = np.sqrt(n)*(u*np.sqrt(p))@u.conj().T
            g = (u*(n*p))@u.conj().T
            lam = np.trace(s).real/n
            gamma = lam*s/n
            assert np.max(abs(np.diag(g)-1)) < 1e-12
            assert abs(np.sum(abs(np.diag(s))**2)/n-value(p)) < 1e-12
            slacks = [float(np.linalg.eigvalsh(
                gamma-np.outer(s[:, j], s[:, j].conj())/n).min()) for j in range(n)]
            assert min(slacks) > -1e-11
            assert abs(np.trace(gamma).real-value(p)) < 1e-12
            # Explicit partial trace of the indexed pure-state amplitudes.
            index = s.T@s.conj()/n
            measured = np.diag(u.T@index@u.conj()).real
            assert np.max(abs(measured-p)) < 1e-12
            rows.append({'N': n, 'support': int(np.count_nonzero(p)),
                         'value': value(p), 'minimum_dual_slack': min(slacks),
                         'measurement_error': float(max(abs(measured-p)))})
    return rows


def compositions(total, n):
    if n == 1:
        yield (total,)
    else:
        for first in range(total+1):
            for rest in compositions(total-first, n-1):
                yield (first,)+rest


def histogram_checks():
    rows = []
    laws = [(1, 0), (1, 1), (1, 7), (1, 100), (1, 1, 1),
            (1, 2, 9), (0, 1, 3), (1, 1, 1, 1)]
    for weights in laws:
        p = [mp.mpf(w)/sum(weights) for w in weights]
        n = len(p)
        truth = sum(mp.sqrt(x) for x in p)**2/n
        for m in [4, 9, 16]:
            total = mp.mpf(0)
            expected_h2 = mp.mpf(0)
            expected_error2 = mp.mpf(0)
            bad_probability = mp.mpf(0)
            alpha = mp.mpf('0.2')
            for counts in compositions(m, n):
                prob = mp.mpf(factorial(m))
                for x, c in zip(p, counts):
                    prob *= x**c/factorial(c)
                if not prob:
                    continue
                q = [mp.mpf(c)/m for c in counts]
                h2 = sum((mp.sqrt(x)-mp.sqrt(y))**2 for x, y in zip(p, q))
                err = abs(sum(mp.sqrt(x) for x in q)**2/n-truth)
                assert err**2 <= h2+mp.mpf('1e-70')
                # Integer-square-root evaluation independently checks the
                # manuscript's dyadic precision prescription.
                chi = mp.mpf('0.001')
                bits = int(mp.ceil(mp.log(4*n/chi, 2)))
                roots = [isqrt((c << (2*bits))//m) for c in counts]
                evaluated = mp.mpf(sum(roots)**2)/(n*(1 << (2*bits)))
                exact_empirical = sum(mp.sqrt(x) for x in q)**2/n
                assert abs(evaluated-exact_empirical) < chi
                total += prob
                expected_h2 += prob*h2
                expected_error2 += prob*err**2
                bad_probability += prob*(err > alpha)
            bound = mp.mpf(sum(x > 0 for x in p)-1)/m
            assert abs(total-1) < mp.mpf('1e-70')
            assert expected_h2 <= bound+mp.mpf('1e-70')
            assert bad_probability <= bound/alpha**2+mp.mpf('1e-70')
            rows.append({'weights': weights, 'block_size': m,
                         'expected_Hellinger_squared': float(expected_h2),
                         'support_bound': float(bound),
                         'expected_value_error_squared': float(expected_error2),
                         'failure_probability_at_0_2': float(bad_probability)})
    return rows


def robustness_checks():
    rows = []
    for _ in range(24):
        z = RNG.uniform(0, .95)*np.exp(1j*RNG.uniform(-np.pi, np.pi))
        k = np.array([[1, z], [z.conjugate(), 1]])/2
        u = np.diag(np.exp(1j*RNG.normal(size=2)))@fourier(2)
        p = np.diag(u.conj().T@k@u).real
        dephased = (u*p)@u.conj().T
        noise = np.array([[.9, 0], [0, .1]])
        actual = .93*k+.07*noise
        ideal = np.diag(u.T@actual.T@u.conj()).real
        angle = .04
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        r = np.diag(rotation@u.T@actual.T@u.conj()@rotation.T).real
        tau, nu = bures(k, dephased), bures(k, actual)
        xi = float(abs(ideal-r).sum()/2)
        truth = (1+np.sqrt(1-abs(z)**2))/2
        radius = tau+nu+np.sqrt(2*xi)
        assert abs(truth-value(r)) <= radius+1e-10
        rows.append({'symmetry_defect': tau, 'preparation_Bures': nu,
                     'measurement_TV': xi, 'bias': abs(truth-value(r)),
                     'certified_bias_radius': radius})
    return rows


def budgets():
    rows = []
    for n in [2, 4, 8, 32]:
        for eps in [mp.mpf(1)/64, mp.mpf(1)/256, mp.mpf(1)/1024]:
            for delta in [mp.mpf(1)/4, mp.mpf(1)/20, mp.mpf('1e-6')]:
                m0 = int(mp.ceil(4*(n-1)/eps**2))
                b = int(mp.ceil(8*mp.log(1/delta)))
                b += 1-b%2
                assert mp.mpf(n-1)/(m0*eps**2) <= mp.mpf(1)/4
                tail = sum(mp.mpf(comb(b, j))*mp.mpf('.25')**j*
                           mp.mpf('.75')**(b-j) for j in range((b+1)//2, b+1))
                assert tail <= mp.exp(-mp.mpf(b)/8) <= delta
                a0, a1 = mp.mpf(1)/8, mp.mpf(1)/8+8*eps
                gap = mp.sqrt(a1*(1-a1))-mp.sqrt(a0*(1-a0))
                kl = a0*mp.log(a0/a1)+(1-a0)*mp.log((1-a0)/(1-a1))
                assert gap > 2*eps
                assert kl <= mp.mpf(4096)/7*eps**2
                # The even-dimensional family has the same value and KL.
                p0 = [2*a0/n]*(n//2)+[2*(1-a0)/n]*(n//2)
                p1 = [2*a1/n]*(n//2)+[2*(1-a1)/n]*(n//2)
                assert abs(sum(x*mp.log(x/y) for x, y in zip(p0, p1))-kl) < mp.mpf('1e-70')
                target_kl = (1-2*delta)*mp.log((1-delta)/delta)
                rows.append({'N': n, 'epsilon': float(eps), 'delta': float(delta),
                             'copies': b*m0, 'scan_controlled_calls': n*b*m0,
                             'blocks': b, 'median_failure_bound': float(tail),
                             'value_gap_over_epsilon': float(gap/eps),
                             'lower_bound': float(7*target_kl/(4096*eps**2))})
    return rows


def adverse_checks():
    a = .25
    g = np.array([[1, a, -a], [a, 1, 0], [-a, 0, 1.]])
    u = fourier(3)
    p = np.diag(u.conj().T@g@u).real/3
    assert np.linalg.eigvalsh(g).min() > 0
    assert max(abs(p-1/3)) < 1e-12
    helstrom_upper = (2+np.sqrt(1-a*a))/3
    assert value(p) > helstrom_upper+1e-3
    # Conjugation matters for a flat basis with nontrivial row phases.
    u = np.diag(np.exp(1j*np.array([.2, .7, 1.4])))@fourier(3)
    p = np.array([.1, .2, .7])
    k = (u*p)@u.conj().T
    wrong = np.diag(u.conj().T@k.T@u).real
    assert abs(value(wrong)-value(p)) > 1e-3
    # Full-rank measurement error can change the value with no shot noise.
    altered = np.array([1/3]*3)
    assert abs(value(altered)-value(p)) > .05
    # The value/entropy accuracy contracts are different near small values.
    p0, p1 = [1., 0., 0., 0.], [.99, .01, 0., 0.]
    assert abs(log(4*value(p1))-log(4*value(p0))) > abs(value(p1)-value(p0))
    return {'uncertified_symmetry_rejected': True,
            'untransposed_index_rotation_rejected': True,
            'omitted_measurement_bias_rejected': True,
            'value_accuracy_is_not_entropy_accuracy': True,
            'collision_value_upper_bound': float(helstrom_upper)}


def main():
    result = {'audit_id': 'AQ_SDP_DIRECT_VALUE_v1',
              'physical_witness_rows': witnesses(),
              'exhaustive_histogram_rows': histogram_checks(),
              'robustness_rows': robustness_checks(),
              'accuracy_confidence_rows': budgets(),
              'adverse_controls': adverse_checks(),
              'all_pass': True,
              'scope': 'Finite consistency checks; no independent review, dimension optimality, or hardware implementation.'}
    here = Path(__file__).resolve().parent
    out = (here/'results' if (here/'results').is_dir() else here)/'AQ_SDP_DIRECT_VALUE_v1_RESULT.json'
    out.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'all_pass': True, 'physical_witness_rows': 40,
                      'exhaustive_histogram_rows': 24, 'robustness_rows': 24,
                      'accuracy_confidence_rows': 36, 'adverse_controls': 4}))


if __name__ == '__main__':
    main()
