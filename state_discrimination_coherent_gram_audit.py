#!/usr/bin/env python3
"""Finite checks of diagonal-drift stability and the coherent copy ledger.

This does not implement the collective tomography POVM. Its tail bound is
an explicitly cited external theorem. The adaptive alternative uses a
squared-fidelity convention, whose conversion is checked in each budget row. Exact two-outcome optima and dense
small preparation circuits independently check the new value interface.
"""
from pathlib import Path
import json
import mpmath as mp
import numpy as np

mp.mp.dps = 80
RNG = np.random.default_rng(2026090507)


def root(a):
    w, v = np.linalg.eigh((a+a.conj().T)/2)
    assert min(w) > -1e-12
    return (v*np.sqrt(np.maximum(w, 0))) @ v.conj().T


def bures(a, b):
    sa, sb = root(a), root(b)
    # Procrustes distance avoids cancellation in 2-2F for nearby states.
    u, _, vh = np.linalg.svd(sa @ sb)
    return float(np.linalg.norm(sa-u@vh@sb))


def density(n, rank):
    x = RNG.normal(size=(n, rank))+1j*RNG.normal(size=(n, rank))
    a = x@x.conj().T
    return a/np.trace(a).real


def optimum_two(a, rewards):
    s = root(a)
    first = s@np.diag(rewards[0])@s
    second = s@np.diag(rewards[1])@s
    w = np.linalg.eigvalsh(first-second)
    return float(np.trace(second).real+np.maximum(w, 0).sum())


def drift_rows():
    rows = []
    for n in [2, 3, 5, 8]:
        for k in range(20):
            a, b = density(n, 1+k%n), density(n, 1+(3*k)%n)
            if k%4 == 0:
                # Include zero estimated priors without any normalization by them.
                b[-1, :] = 0
                b[:, -1] = 0
                b /= np.trace(b).real
            rewards = RNG.normal(size=(2, n))*3
            center = (rewards.max(axis=0)+rewards.min(axis=0))/2
            centered = rewards-center
            ranges = np.ptp(rewards, axis=0)
            q, p = np.diag(a).real, np.diag(b).real
            va, vb = optimum_two(a, centered), optimum_two(b, centered)
            distance = bures(a, b)
            coefficient = .5*np.linalg.norm(ranges*(np.sqrt(q)+np.sqrt(p)))
            assert coefficient <= ranges.max()+1e-10
            assert abs(va-vb) <= coefficient*distance+1e-9
            actual = optimum_two(a, rewards)
            assert abs(actual-(q@center+va)) < 1e-9
            estimate = q@center+vb
            assert abs(actual-estimate) <= ranges.max()*distance+1e-9
            rows.append({'N': n, 'zero_estimated_prior': bool(p[-1] == 0),
                         'value_error': abs(actual-estimate),
                         'Bures_bound': float(coefficient*distance)})
    return rows


def unitary(d):
    x = RNG.normal(size=(d, d))+1j*RNG.normal(size=(d, d))
    return np.linalg.qr(x)[0]


def preparation_rows():
    rows = []
    for n in [2, 3, 5]:
        size = 1 << (n-1).bit_length()
        for d in [2, 4, 8]:
            us = [unitary(d) for _ in range(n)]
            q = RNG.random(n)
            q /= q.sum()
            omega = np.zeros((size, d), dtype=complex)
            omega[:n, 0] = np.sqrt(q)
            calls = 0
            for j, u in enumerate(us):
                for index in range(size):
                    # Equality flag compute, controlled target, uncompute.
                    if index == j:
                        omega[index] = u@omega[index]
                calls += 1
            psi = np.array([u[:, 0] for u in us])
            gram = psi.conj()@psi.T
            k = np.sqrt(q[:, None]*q[None, :])*gram
            marginal = omega@omega.conj().T
            assert np.max(np.abs(marginal[:n, :n]-k.T)) < 1e-13
            assert np.linalg.norm(marginal[n:, :]) == 0
            assert calls == n
            assert np.linalg.norm(k-k.T) > 1e-5  # genuinely complex control
            z = RNG.normal(size=omega.shape)+1j*RNG.normal(size=omega.shape)
            z[n:, :] = 0
            z -= omega*np.vdot(omega, z)
            z /= np.linalg.norm(z)
            noisy = np.sqrt(1-.01**2)*omega+.01*z
            noisy_k = (noisy@noisy.conj().T)[:n, :n].T
            vector_error = float(np.linalg.norm(noisy-omega))
            marginal_error = bures(k, noisy_k)
            assert marginal_error <= vector_error+1e-7
            rows.append({'N': n, 'source_dimension': d, 'controlled_calls': calls,
                         'marginal_identity_error': float(np.max(np.abs(marginal[:n,:n]-k.T))),
                         'preparation_vector_error': vector_error,
                         'marginal_Bures_error': marginal_error})
    return rows


def budget_rows():
    rows = []
    for n in [2, 3, 8, 32]:
        for rank in [1, n]:
            for ratio in [mp.mpf(1), mp.mpf(1)/4, mp.mpf(1)/64]:
                eta = ratio**2/8
                for delta in [mp.mpf(1)/4, mp.mpf(1)/1000]:
                    a, b = 3*n*rank, mp.log(1/delta)
                    m = int(mp.ceil((a*mp.log(4*a/eta)+b)/eta))
                    log_failure = a*mp.log(m+1)-2*m*eta
                    assert log_failure <= mp.log(delta)
                    assert 2*(1-mp.sqrt(1-eta)) <= 2*eta
                    radius = ratio/4+mp.sqrt(2*eta)+ratio/4
                    assert abs(radius-ratio) < mp.mpf('1e-75')
                    rows.append({'N': n, 'actual_rank_bound': rank,
                                 'epsilon_over_range': float(ratio),
                                 'delta': float(delta), 'index_copies': m,
                                 'scan_controlled_calls': n*m,
                                 'log_failure_bound': float(log_failure),
                                 'adaptive_fidelity_radius': float(mp.sqrt(2*(1-mp.sqrt(1-eta))))})
    return rows


def adverse_controls():
    a, b = np.diag([.9, .1]), np.diag([.1, .9])
    reward = np.array([[100., 0.], [100., 0.]])
    # Zero reward ranges do not bound raw values when the priors drift.
    assert abs(optimum_two(a, reward)-optimum_two(b, reward)) > 79
    assert optimum_two(a, reward-reward[0]) == 0
    assert optimum_two(b, reward-reward[0]) == 0
    pure, mixed = np.diag([1., 0.]), np.diag([.9, .1])
    eta = 1-np.sqrt(.9)
    assert abs(bures(pure, mixed)**2-2*eta) < 1e-12
    assert bures(pure, mixed) > np.sqrt(eta)
    # Small preparation error can increase rank: charge the actual marginal.
    base = np.zeros((3, 4))
    base[:, 0] = 1/np.sqrt(3)
    error = np.zeros((3, 4))
    error[:, 1:] = np.eye(3)/np.sqrt(3)
    noisy = np.sqrt(1-1e-6)*base+1e-3*error
    assert np.linalg.matrix_rank(base@base.T) == 1
    assert np.linalg.matrix_rank(noisy@noisy.T) == 3
    return {'uncentered_diagonal_drift_rejected': True,
            'missing_sqrt_two_in_root_fidelity_conversion_rejected': True,
            'ideal_rank_used_after_preparation_noise_rejected': True,
            'missing_marginal_transpose_rejected_in_preparation_rows': True,
            'free_indexing_rejected_by_scan_call_counts': True}


def main():
    packet = {
        'status': 'coherent_Gram_value_diagonal_drift_and_copy_budget_checked',
        'scope': 'Finite interface checks; collective tomography tail inherited '
                 'from arXiv:1508.01797v2 Eq.(14). No tomography POVM '
                 'implementation, efficient runtime, or advantage claim.',
        'drifting_diagonal_value_rows': drift_rows(),
        'coherent_preparation_rows': preparation_rows(),
        'copy_budget_rows': budget_rows(),
        'adverse_controls': adverse_controls(),
    }
    folder = Path(__file__).resolve().parent
    result_dir = folder/'results' if folder.name == 'numerics' else folder
    result_dir.mkdir(exist_ok=True, parents=True)
    (result_dir/'AQ_SDP_COHERENT_GRAM_v1_RESULT.json').write_text(
        json.dumps(packet, indent=2)+'\n')
    print(packet['status'])
    print('80 drifting-diagonal value rows; 9 complex preparation rows; '
          '48 copy budgets; 5 adverse controls')


if __name__ == '__main__':
    main()
