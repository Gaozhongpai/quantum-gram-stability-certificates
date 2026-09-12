#!/usr/bin/env python3
"""Audit prior-weighted Bayes-value stability for sampled Gram matrices.

The packet checks the analytic ingredients of AQ-SDP-GRAM-BAYES: canonical
prior-weighted ensembles, Bures alignment, measurement/reward contraction,
the Powers--Størmer trace-norm bound, exact two-state Bayes optima, explicit
and nearest-correlation repairs, arbitrary-prior sampling ledgers, and
constructive reduced primal/dual intervals.  It makes no physical-POVM,
runtime-advantage, sample-optimality, or C2/C3 claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_RESULT = (
    SCRIPT_DIRECTORY / "results/AQ_SDP_BAYES_GRAM_STABILITY_v1_RESULT.json"
)
RESULT = (
    REPOSITORY_RESULT
    if REPOSITORY_RESULT.parent.is_dir()
    else SCRIPT_DIRECTORY / "AQ_SDP_BAYES_GRAM_STABILITY_v1_RESULT.json"
)
CANDIDATE_ID = "AQ-SDP-GRAM-BAYES-v1"
STATUS = "Bures_Bayes_stability_nearest_correlation_N3_sampling_closed"
TOL = 2e-10


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conj().T) / 2


def psd_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(hermitian(matrix))
    values = np.maximum(values, 0.0)
    return hermitian((vectors * np.sqrt(values)) @ vectors.conj().T)


def psd_projection(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(hermitian(matrix))
    values = np.maximum(values, 0.0)
    return hermitian((vectors * values) @ vectors.conj().T)


def psd_inverse_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(hermitian(matrix))
    if float(np.min(values)) <= 1e-12:
        raise AssertionError("registered POVM normalizer must be positive definite")
    return hermitian((vectors * (1 / np.sqrt(values))) @ vectors.conj().T)


def trace_norm_hermitian(matrix: np.ndarray) -> float:
    return float(np.sum(np.abs(np.linalg.eigvalsh(hermitian(matrix)))))


def correlation_matrix(count: int, rng: np.random.Generator) -> np.ndarray:
    dimension = max(2, count // 2 + 1)
    states = rng.normal(size=(dimension, count)) + 1j * rng.normal(
        size=(dimension, count)
    )
    states /= np.linalg.norm(states, axis=0, keepdims=True)
    return hermitian(states.conj().T @ states)


def well_conditioned_correlation_matrix(
    count: int, rng: np.random.Generator, mix: float = 0.55
) -> np.ndarray:
    """Return a correlation matrix with a deterministic positive gap floor."""

    raw = correlation_matrix(count, rng)
    return hermitian((1 - mix) * np.eye(count) + mix * raw)


def bounded_noisy_hermitian(
    gram: np.ndarray, eta: float, rng: np.random.Generator
) -> np.ndarray:
    count = gram.shape[0]
    noisy = gram.copy()
    for row in range(count):
        for column in range(row + 1, count):
            radius = eta * float(rng.random())
            angle = 2 * math.pi * float(rng.random())
            error = radius * complex(math.cos(angle), math.sin(angle))
            noisy[row, column] += error
            noisy[column, row] = noisy[row, column].conjugate()
    np.fill_diagonal(noisy, 1.0)
    return hermitian(noisy)


def shift_repair(unit_diagonal_hermitian: np.ndarray, eta: float):
    count = unit_diagonal_hermitian.shape[0]
    tau = (count - 1) * eta
    repaired = hermitian(
        (unit_diagonal_hermitian + tau * np.eye(count)) / (1 + tau)
    )
    return repaired, tau


def nearest_correlation(
    unit_diagonal_hermitian: np.ndarray,
    tolerance: float = 2e-13,
    maximum_iterations: int = 20000,
) -> tuple[np.ndarray, int]:
    """Dykstra projection onto the complex correlation-matrix elliptope."""

    current = hermitian(unit_diagonal_hermitian.copy())
    np.fill_diagonal(current, 1.0)
    correction = np.zeros_like(current)
    for iteration in range(1, maximum_iterations + 1):
        residual = current - correction
        positive = psd_projection(residual)
        correction = positive - residual
        updated = positive.copy()
        np.fill_diagonal(updated, 1.0)
        updated = hermitian(updated)
        scale = max(1.0, float(np.linalg.norm(updated, ord="fro")))
        if float(np.linalg.norm(updated - current, ord="fro")) <= tolerance * scale:
            return updated, iteration
        current = updated
    raise AssertionError("nearest-correlation projection did not converge")


def canonical_weighted_states(gram: np.ndarray, priors: np.ndarray) -> np.ndarray:
    qroot = np.diag(np.sqrt(priors))
    weighted_gram = hermitian(qroot @ gram @ qroot)
    subnormalized = psd_root(weighted_gram)
    return subnormalized / np.sqrt(priors)[None, :]


def bures_aligned_weighted_states(
    gram: np.ndarray, other: np.ndarray, priors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    qroot = np.diag(np.sqrt(priors))
    left_root = psd_root(hermitian(qroot @ gram @ qroot))
    right_root = psd_root(hermitian(qroot @ other @ qroot))
    left_singular, _, right_adjoint = np.linalg.svd(right_root @ left_root)
    alignment = right_adjoint.conj().T @ left_singular.conj().T
    aligned_right = alignment @ right_root
    distance_squared = float(np.linalg.norm(left_root - aligned_right, ord="fro") ** 2)
    left_states = left_root / np.sqrt(priors)[None, :]
    right_states = aligned_right / np.sqrt(priors)[None, :]
    return left_states, right_states, distance_squared


def bures_distance_squared(weighted_left: np.ndarray, weighted_right: np.ndarray) -> float:
    left_root = psd_root(weighted_left)
    middle_root = psd_root(hermitian(left_root @ weighted_right @ left_root))
    value = float(
        np.real(np.trace(weighted_left) + np.trace(weighted_right) - 2 * np.trace(middle_root))
    )
    return max(0.0, value)


def random_povm(dimension: int, outcomes: int, rng: np.random.Generator):
    raw = []
    for _ in range(outcomes):
        matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
            size=(dimension, dimension)
        )
        raw.append(matrix @ matrix.conj().T)
    normalizer = psd_inverse_root(sum(raw))
    effects = [hermitian(normalizer @ item @ normalizer) for item in raw]
    residual = np.linalg.norm(sum(effects) - np.eye(dimension), ord=2)
    if residual > 2e-10:
        raise AssertionError("POVM normalization failed")
    return effects


def pure_trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    overlap = np.vdot(left, right)
    return math.sqrt(max(0.0, 1.0 - min(1.0, abs(overlap) ** 2)))


def reward_value(
    states: np.ndarray,
    priors: np.ndarray,
    rewards: np.ndarray,
    effects: list[np.ndarray],
) -> float:
    value = 0.0
    for state_index in range(states.shape[1]):
        state = states[:, state_index]
        for outcome, effect in enumerate(effects):
            probability = float(np.real(np.vdot(state, effect @ state)))
            value += priors[state_index] * rewards[outcome, state_index] * probability
    return value


def two_outcome_optimum(
    states: np.ndarray, priors: np.ndarray, rewards: np.ndarray
) -> float:
    base = float(np.dot(priors, rewards[1, :]))
    decision = np.zeros((states.shape[0], states.shape[0]), dtype=complex)
    for index in range(states.shape[1]):
        state = states[:, index]
        delta = rewards[0, index] - rewards[1, index]
        decision += priors[index] * delta * np.outer(state, state.conj())
    positive_part = float(
        np.sum(np.maximum(np.linalg.eigvalsh(hermitian(decision)), 0.0))
    )
    return base + positive_part


def bayes_bound(
    gram: np.ndarray,
    other: np.ndarray,
    priors: np.ndarray,
    rewards: np.ndarray,
) -> tuple[float, float, float]:
    qroot = np.diag(np.sqrt(priors))
    weighted_trace_distance = trace_norm_hermitian(
        qroot @ (gram - other) @ qroot
    )
    reward_ranges = np.max(rewards, axis=0) - np.min(rewards, axis=0)
    reward_rms = math.sqrt(float(np.dot(priors, reward_ranges**2)))
    bound = reward_rms * math.sqrt(weighted_trace_distance)
    return bound, reward_rms, weighted_trace_distance


def eta_for_uniform_success_error(count: int, epsilon: float) -> float:
    # Nearest-correlation repair gives error at most
    # sqrt(sqrt(N-1) eta).
    return epsilon**2 / math.sqrt(count - 1)


def eta_for_bayes_error(
    count: int,
    epsilon: float,
    reward_range_rms: float,
    maximum_prior: float,
) -> float | None:
    """Common entry radius for the arbitrary-prior finite-reward ledger."""

    if reward_range_rms == 0:
        return None
    denominator = reward_range_rms**2 * math.sqrt(
        maximum_prior * count * (count - 1)
    )
    return epsilon**2 / denominator


def two_outcome_reduced_certificate(
    gram: np.ndarray, priors: np.ndarray, rewards: np.ndarray
) -> dict[str, float]:
    """Construct a feasible optimal primal and shifted feasible dual for L=2.

    Positive definiteness is imposed only for this finite certificate
    construction, not for the paper's stability theorem.
    """

    if rewards.shape != (2, gram.shape[0]):
        raise AssertionError("two-outcome certificate requires a 2 x N reward table")
    count = gram.shape[0]
    root = psd_root(gram)
    inverse_root = psd_inverse_root(gram)
    objectives = [np.diag(priors * rewards[index, :]) for index in range(2)]
    contrast = hermitian(root @ (objectives[0] - objectives[1]) @ root)
    values, vectors = np.linalg.eigh(contrast)
    positive_values = np.maximum(values, 0.0)
    positive_part = hermitian((vectors * positive_values) @ vectors.conj().T)
    positive_projector = hermitian(
        (vectors * (values > 0).astype(float)) @ vectors.conj().T
    )

    primal_zero = hermitian(root @ positive_projector @ root)
    primal_one = hermitian(gram - primal_zero)
    lower = float(
        np.real(
            np.trace(objectives[0] @ primal_zero)
            + np.trace(objectives[1] @ primal_one)
        )
    )

    dual_candidate = hermitian(
        objectives[1] + inverse_root @ positive_part @ inverse_root
    )
    raw_floor = min(
        float(np.min(np.linalg.eigvalsh(hermitian(dual_candidate - objective))))
        for objective in objectives
    )
    dual_shift = max(0.0, -raw_floor) + 5e-12
    dual = hermitian(dual_candidate + dual_shift * np.eye(count))
    upper = float(np.real(np.trace(dual @ gram)))
    dual_floor = min(
        float(np.min(np.linalg.eigvalsh(hermitian(dual - objective))))
        for objective in objectives
    )
    exact_formula = float(
        np.real(np.trace(objectives[1] @ gram)) + np.sum(positive_values)
    )
    return {
        "primal_lower": lower,
        "dual_upper": upper,
        "exact_reduced_formula": exact_formula,
        "interval_width": upper - lower,
        "primal_sum_residual": float(
            np.linalg.norm(primal_zero + primal_one - gram, ord="fro")
        ),
        "primal_minimum_eigenvalue": min(
            float(np.min(np.linalg.eigvalsh(primal_zero))),
            float(np.min(np.linalg.eigvalsh(primal_one))),
        ),
        "dual_feasibility_floor": dual_floor,
        "dual_identity_shift": dual_shift,
    }


def gap_refinement_checks():
    rng = np.random.default_rng(20260908)
    rows = []
    for n in (2, 3, 4):
        for floor in (0.05, 0.2, 0.5, 0.8):
            for _ in range(4):
                g = floor*np.eye(n)+(1-floor)*correlation_matrix(n, rng)
                h = floor*np.eye(n)+(1-floor)*correlation_matrix(n, rng)
                x = psd_root(g)-psd_root(h)
                assert np.linalg.norm(psd_root(g)@x+x@psd_root(h)-(g-h), 'fro') < TOL
                gap_sum = math.sqrt(np.linalg.eigvalsh(g)[0])+math.sqrt(np.linalg.eigvalsh(h)[0])
                assert np.linalg.norm(x, 'fro') <= np.linalg.norm(g-h, 'fro')/gap_sum + TOL
                priors = rng.random(n); priors /= sum(priors)
                rewards = rng.normal(size=(2,n))
                delta = math.sqrt(np.dot(priors, np.ptp(rewards, axis=0)**2))
                value_gap = abs(two_outcome_optimum(psd_root(g),priors,rewards)
                                -two_outcome_optimum(psd_root(h),priors,rewards))
                bound = delta*math.sqrt(max(priors))*np.linalg.norm(g-h,'fro')/gap_sum
                assert value_gap <= bound+TOL
                eta = 0.015
                noisy = bounded_noisy_hermitian(g,eta,rng)
                projected, _ = nearest_correlation((noisy-floor*np.eye(n))/(1-floor))
                repaired = floor*np.eye(n)+(1-floor)*projected
                assert np.linalg.eigvalsh(repaired)[0] >= floor-TOL
                assert np.linalg.norm(repaired-g,'fro') <= math.sqrt(n*(n-1))*eta+TOL
                rows.append({'N':n, 'floor':floor, 'value_gap':value_gap, 'radius':bound})
    budgets = []
    for n in (2,3,5):
        for floor in (0.1,0.3,0.8):
            for epsilon in (0.01,0.03,0.1,0.2):
                eta = 2*math.sqrt(floor)*epsilon/math.sqrt(n-1)
                assert abs(math.sqrt(n-1)*eta/(2*math.sqrt(floor))-epsilon) < TOL
                assert abs(4/eta**2-(n-1)/(floor*epsilon**2)) < 1e-7
                budgets.append({'N':n, 'floor':floor, 'epsilon':epsilon})
    # A false gap promise biases even exact overlap data at the singular boundary.
    g = np.ones((2,2)); floor = 0.2
    projected, _ = nearest_correlation((g-floor*np.eye(2))/(1-floor))
    h = floor*np.eye(2)+(1-floor)*projected
    assert np.linalg.norm(h-g,'fro') > 0.1
    value_gap = math.sqrt(1-h[0,1]**2)/2
    assert value_gap > 0.2
    return {'matrix_and_value_rows':rows, 'sampling_rows':budgets,
            'false_gap_with_exact_data_rejected':True,
            'singular_rank_fourth_power_claim_unchanged':True}


def main() -> int:
    rng = np.random.default_rng(20260829)

    fixed_measurement_rows = []
    for count in (2, 3, 4, 6, 8):
        for outcomes in (2, 3):
            gram = correlation_matrix(count, rng)
            other = correlation_matrix(count, rng)
            priors = rng.dirichlet(np.ones(count))
            rewards = rng.uniform(-1.0, 2.0, size=(outcomes, count))
            left_states, right_states, aligned_distance = bures_aligned_weighted_states(
                gram, other, priors
            )
            effects = random_povm(count, outcomes, rng)
            left_value = reward_value(left_states, priors, rewards, effects)
            right_value = reward_value(right_states, priors, rewards, effects)
            value_difference = abs(left_value - right_value)
            reward_ranges = np.max(rewards, axis=0) - np.min(rewards, axis=0)
            trace_distances = np.array(
                [
                    pure_trace_distance(left_states[:, index], right_states[:, index])
                    for index in range(count)
                ]
            )
            operational_bound = float(np.dot(priors * reward_ranges, trace_distances))
            theorem_bound, reward_rms, weighted_trace = bayes_bound(
                gram, other, priors, rewards
            )
            weighted_left = np.diag(np.sqrt(priors)) @ gram @ np.diag(
                np.sqrt(priors)
            )
            weighted_right = np.diag(np.sqrt(priors)) @ other @ np.diag(
                np.sqrt(priors)
            )
            bures_squared = bures_distance_squared(weighted_left, weighted_right)
            assert value_difference <= operational_bound + TOL
            assert operational_bound <= reward_rms * math.sqrt(bures_squared) + TOL
            assert abs(aligned_distance - bures_squared) <= 2e-8
            assert bures_squared <= weighted_trace + TOL
            fixed_measurement_rows.append(
                {
                    "N": count,
                    "outcomes": outcomes,
                    "fixed_measurement_value_difference": value_difference,
                    "operational_trace_distance_bound": operational_bound,
                    "prior_weighted_Gram_bound": theorem_bound,
                    "reward_range_rms": reward_rms,
                    "weighted_Gram_trace_norm": weighted_trace,
                    "Bures_distance_squared": bures_squared,
                    "Bures_bound": reward_rms * math.sqrt(bures_squared),
                    "all_bounds_hold": True,
                }
            )

    exact_optimum_rows = []
    for _ in range(24):
        gram = correlation_matrix(2, rng)
        other = correlation_matrix(2, rng)
        priors = rng.dirichlet(np.ones(2))
        rewards = rng.uniform(-2.0, 3.0, size=(2, 2))
        left_states = canonical_weighted_states(gram, priors)
        right_states = canonical_weighted_states(other, priors)
        left_optimum = two_outcome_optimum(left_states, priors, rewards)
        right_optimum = two_outcome_optimum(right_states, priors, rewards)
        optimum_difference = abs(left_optimum - right_optimum)
        theorem_bound, reward_rms, weighted_trace = bayes_bound(
            gram, other, priors, rewards
        )
        assert optimum_difference <= theorem_bound + TOL
        exact_optimum_rows.append(
            {
                "N": 2,
                "outcomes": 2,
                "exact_Bayes_optimum_difference": optimum_difference,
                "prior_weighted_Gram_bound": theorem_bound,
                "reward_range_rms": reward_rms,
                "weighted_Gram_trace_norm": weighted_trace,
                "bound_holds": True,
            }
        )

    repair_rows = []
    for count in (2, 3, 4, 6, 8, 12):
        for eta in (1e-4, 1e-3, 1e-2):
            gram = correlation_matrix(count, rng)
            noisy = bounded_noisy_hermitian(gram, eta, rng)
            repaired, tau = shift_repair(noisy, eta)
            priors = rng.dirichlet(np.ones(count))
            rewards = rng.uniform(-1.0, 2.0, size=(3, count))
            theorem_bound, reward_rms, weighted_trace = bayes_bound(
                gram, repaired, priors, rewards
            )
            refined_trace_bound = (
                count * math.sqrt(count - 1) * eta + 2 * (count - 1) * tau
            ) / (1 + tau)
            analytic_bound = reward_rms * math.sqrt(
                float(np.max(priors)) * refined_trace_bound
            )
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(repaired)))
            diagonal_error = float(np.max(np.abs(np.diag(repaired) - 1)))
            assert minimum_eigenvalue >= -TOL
            assert diagonal_error <= TOL
            assert theorem_bound <= analytic_bound + 5e-10
            repair_rows.append(
                {
                    "N": count,
                    "entrywise_complex_error_eta": eta,
                    "shift_tau": tau,
                    "repaired_minimum_eigenvalue": minimum_eigenvalue,
                    "repaired_diagonal_error": diagonal_error,
                    "weighted_Gram_trace_norm": weighted_trace,
                    "refined_unweighted_trace_norm_bound": refined_trace_bound,
                    "instance_Bayes_stability_bound": theorem_bound,
                    "analytic_Bayes_stability_bound": analytic_bound,
                    "all_bounds_hold": True,
                }
            )

    projection_repair_rows = []
    for count in (2, 3, 4, 6, 8, 12):
        for eta in (1e-4, 1e-3, 1e-2):
            gram = correlation_matrix(count, rng)
            noisy = bounded_noisy_hermitian(gram, eta, rng)
            repaired, iterations = nearest_correlation(noisy)
            priors = rng.dirichlet(np.ones(count))
            rewards = rng.uniform(-1.0, 2.0, size=(3, count))
            theorem_bound, reward_rms, weighted_trace = bayes_bound(
                gram, repaired, priors, rewards
            )
            noisy_error = float(np.linalg.norm(noisy - gram, ord="fro"))
            projection_distance = float(np.linalg.norm(repaired - noisy, ord="fro"))
            repair_error = float(np.linalg.norm(repaired - gram, ord="fro"))
            maximum_prior = float(np.max(priors))
            pythagorean_upper_squared = max(
                0.0, noisy_error**2 - projection_distance**2
            )
            instance_schatten_bound = math.sqrt(maximum_prior) * repair_error
            analytic_weighted_trace = (
                math.sqrt(maximum_prior * count * (count - 1)) * eta
            )
            analytic_bound = reward_rms * math.sqrt(analytic_weighted_trace)
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(repaired)))
            diagonal_error = float(np.max(np.abs(np.diag(repaired) - 1)))
            assert minimum_eigenvalue >= -5e-10
            assert diagonal_error <= TOL
            assert projection_distance <= noisy_error + 5e-9
            assert repair_error**2 <= pythagorean_upper_squared + 5e-9
            assert repair_error <= noisy_error + 5e-9
            assert weighted_trace <= instance_schatten_bound + 5e-9
            assert weighted_trace <= analytic_weighted_trace + 5e-9
            assert theorem_bound <= analytic_bound + 5e-9
            projection_repair_rows.append(
                {
                    "N": count,
                    "entrywise_complex_error_eta": eta,
                    "projection_iterations": iterations,
                    "repaired_minimum_eigenvalue": minimum_eigenvalue,
                    "repaired_diagonal_error": diagonal_error,
                    "distance_to_noisy_matrix": projection_distance,
                    "noisy_matrix_error": noisy_error,
                    "repair_error": repair_error,
                    "Pythagorean_repair_error_upper": math.sqrt(
                        pythagorean_upper_squared
                    ),
                    "weighted_Gram_trace_norm": weighted_trace,
                    "instance_Schatten_Holder_bound": instance_schatten_bound,
                    "analytic_weighted_trace_norm_bound": analytic_weighted_trace,
                    "instance_Bayes_stability_bound": theorem_bound,
                    "analytic_Bayes_stability_bound": analytic_bound,
                    "all_bounds_hold": True,
                }
            )

    sampling_rows = []
    for count in (2, 4, 8, 16):
        for epsilon in (0.1, 0.05):
            delta = 0.01
            eta = eta_for_uniform_success_error(count, epsilon)
            quadratures = count * (count - 1)
            shots_per_quadrature = math.ceil(
                4 * math.log(2 * quadratures / delta) / eta**2
            )
            total_hadamard_shots = quadratures * shots_per_quadrature
            certified_probability_bound = math.sqrt(
                math.sqrt(count - 1) * eta
            )
            assert certified_probability_bound <= epsilon * (1 + 1e-12)
            sampling_rows.append(
                {
                    "N": count,
                    "minimum_error_probability_epsilon": epsilon,
                    "global_failure_delta": delta,
                    "complex_entry_error_eta": eta,
                    "quadrature_means": quadratures,
                    "shots_per_quadrature": shots_per_quadrature,
                    "total_hadamard_shots": total_hadamard_shots,
                    "controlled_preparation_or_inverse_calls": 2
                    * total_hadamard_shots,
                    "certified_success_probability_error": certified_probability_bound,
                    "leading_sufficient_scaling": "N^3 epsilon^-4 log(N/delta)",
                    "bound_is_conservative_not_optimal": True,
                }
            )

    general_sampling_rows = []
    for count in (2, 4, 8, 16):
        for epsilon in (0.1, 0.05):
            delta = 0.01
            priors = rng.dirichlet(np.ones(count))
            rewards = rng.uniform(-2.0, 3.0, size=(3, count))
            reward_ranges = np.max(rewards, axis=0) - np.min(rewards, axis=0)
            reward_rms = math.sqrt(float(np.dot(priors, reward_ranges**2)))
            maximum_prior = float(np.max(priors))
            eta = eta_for_bayes_error(
                count, epsilon, reward_rms, maximum_prior
            )
            if eta is None:
                raise AssertionError("registered nonconstant reward unexpectedly has zero range")
            quadratures = count * (count - 1)
            shots_per_quadrature = math.ceil(
                4 * math.log(2 * quadratures / delta) / eta**2
            )
            total_hadamard_shots = quadratures * shots_per_quadrature
            certified_bayes_bound = (
                reward_rms
                * (maximum_prior * count * (count - 1)) ** 0.25
                * math.sqrt(eta)
            )
            assert certified_bayes_bound <= epsilon * (1 + 1e-12)
            general_sampling_rows.append(
                {
                    "N": count,
                    "Bayes_reward_epsilon": epsilon,
                    "global_failure_delta": delta,
                    "reward_range_rms": reward_rms,
                    "maximum_prior": maximum_prior,
                    "complex_entry_error_eta": eta,
                    "quadrature_means": quadratures,
                    "shots_per_quadrature": shots_per_quadrature,
                    "total_hadamard_shots": total_hadamard_shots,
                    "controlled_preparation_or_inverse_calls": 2
                    * total_hadamard_shots,
                    "certified_Bayes_reward_error": certified_bayes_bound,
                    "bound_is_conservative_not_optimal": True,
                }
            )

    zero_count = 4
    zero_priors = rng.dirichlet(np.ones(zero_count))
    zero_column_rewards = rng.uniform(-2.0, 3.0, size=zero_count)
    zero_rewards = np.vstack((zero_column_rewards, zero_column_rewards))
    zero_left = correlation_matrix(zero_count, rng)
    zero_right = correlation_matrix(zero_count, rng)
    zero_left_value = two_outcome_optimum(
        canonical_weighted_states(zero_left, zero_priors),
        zero_priors,
        zero_rewards,
    )
    zero_right_value = two_outcome_optimum(
        canonical_weighted_states(zero_right, zero_priors),
        zero_priors,
        zero_rewards,
    )
    zero_formula = float(np.dot(zero_priors, zero_column_rewards))
    assert eta_for_bayes_error(zero_count, 0.1, 0.0, max(zero_priors)) is None
    assert abs(zero_left_value - zero_formula) <= TOL
    assert abs(zero_right_value - zero_formula) <= TOL
    zero_reward_range_row = {
        "N": zero_count,
        "reward_range_rms": 0.0,
        "first_Gram_value": zero_left_value,
        "second_Gram_value": zero_right_value,
        "Gram_independent_formula": zero_formula,
        "required_overlap_shots_for_value": 0,
        "all_checks_hold": True,
    }

    primal_dual_interval_rows = []
    for count in (2, 3, 4, 6):
        for eta in (1e-4, 1e-3, 1e-2):
            gram = well_conditioned_correlation_matrix(count, rng)
            noisy = bounded_noisy_hermitian(gram, eta, rng)
            repaired, iterations = nearest_correlation(noisy)
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(repaired)))
            if minimum_eigenvalue <= 1e-8:
                raise AssertionError("registered solver-certificate Gram lost its gap")
            priors = rng.dirichlet(np.ones(count))
            rewards = rng.uniform(-2.0, 3.0, size=(2, count))
            certificate = two_outcome_reduced_certificate(
                repaired, priors, rewards
            )
            exact_state_value = two_outcome_optimum(
                canonical_weighted_states(repaired, priors),
                priors,
                rewards,
            )
            assert certificate["primal_sum_residual"] <= 5e-10
            assert certificate["primal_minimum_eigenvalue"] >= -5e-10
            assert certificate["dual_feasibility_floor"] >= -5e-10
            assert certificate["primal_lower"] <= exact_state_value + 5e-9
            assert exact_state_value <= certificate["dual_upper"] + 5e-9
            assert (
                abs(certificate["exact_reduced_formula"] - exact_state_value)
                <= 5e-9
            )
            primal_dual_interval_rows.append(
                {
                    "N": count,
                    "entrywise_complex_error_eta": eta,
                    "projection_iterations": iterations,
                    "repaired_minimum_eigenvalue": minimum_eigenvalue,
                    "exact_state_space_value": exact_state_value,
                    **certificate,
                    "all_checks_hold": True,
                }
            )

    payload = {
        "gap_sensitive_refinement": gap_refinement_checks(),
        "candidate_id": CANDIDATE_ID,
        "status": STATUS,
        "theorem": {
            "reward_promise": "finite rewards R_ij with column ranges Delta_j",
            "priors": "q_j>0 and sum_j q_j=1; Q=diag(q)",
            "bound": "|Val_R,q(G)-Val_R,q(H)| <= sqrt(sum_j q_j Delta_j^2) sqrt(||Q^1/2(G-H)Q^1/2||_1)",
            "stronger_Bures_bound": "|Val_R,q(G)-Val_R,q(H)| <= sqrt(sum_j q_j Delta_j^2) d_B(Q^1/2 G Q^1/2,Q^1/2 H Q^1/2)",
            "uniform_minimum_error": "|Psucc(G)-Psucc(H)| <= sqrt(||G-H||_1/N)",
        },
        "repair": {
            "explicit_construction": "H_shift=(B+(N-1)eta I)/(1+(N-1)eta)",
            "sharp_construction": "H_star=argmin{||C-B||_F:C PSD, diag(C)=1}",
            "metric_projection_Pythagorean_bound": "||H_star-G||_F^2 <= ||B-G||_F^2-||B-H_star||_F^2 <= N(N-1)eta^2",
            "prior_weighted_trace_bound": "||Q^1/2(H_star-G)Q^1/2||_1 <= sqrt(q_max N(N-1)) eta",
            "nearest_repair_uniform_minimum_error_bound": "sqrt(sqrt(N-1) eta)",
        },
        "sampling_contract": {
            "quadratures": "N(N-1) real/imaginary means",
            "shots_per_quadrature": "ceil(4 eta^-2 ln(2N(N-1)/delta))",
            "controlled_calls_per_shot": 2,
            "general_eta_choice": "eta=epsilon^2/(Delta_q^2 sqrt(q_max N(N-1))) when Delta_q>0",
            "zero_reward_range": "Delta_q=0 makes the value Gram-independent and requires zero overlap shots",
            "uniform_eta_choice": "eta=epsilon^2/sqrt(N-1)",
            "uniform_total_shot_coefficient": 4,
            "leading_sufficient_scaling": "N^3 epsilon^-4 log(N/delta)",
        },
        "solver_certificate": {
            "primal": "maximize sum_i Tr(D_i W_i) subject to W_i PSD and sum_i W_i=H",
            "dual": "minimize Tr(YH) subject to Y>=D_i for every i",
            "dual_repair": "Y=Y_tilde+alpha I, alpha=max_i max(0,-lambda_min(Y_tilde-D_i))",
            "floating_point_output_alone_is_certified": False,
        },
        "fixed_measurement_rows": fixed_measurement_rows,
        "exact_two_outcome_optimum_rows": exact_optimum_rows,
        "repair_rows": repair_rows,
        "projection_repair_rows": projection_repair_rows,
        "sampling_rows": sampling_rows,
        "general_sampling_rows": general_sampling_rows,
        "zero_reward_range_row": zero_reward_range_row,
        "primal_dual_interval_rows": primal_dual_interval_rows,
        "aggregate": {
            "fixed_measurement_rows": len(fixed_measurement_rows),
            "exact_two_outcome_optimum_rows": len(exact_optimum_rows),
            "repair_rows": len(repair_rows),
            "projection_repair_rows": len(projection_repair_rows),
            "sampling_rows": len(sampling_rows),
            "general_sampling_rows": len(general_sampling_rows),
            "zero_reward_range_rows": 1,
            "primal_dual_interval_rows": len(primal_dual_interval_rows),
            "all_rows_pass": True,
        },
        "scope": {
            "physical_POVM_output_closed": False,
            "certified_SDP_value_interval_composition_closed": True,
            "constructive_primal_dual_interval_closed": True,
            "arbitrary_prior_finite_reward_sampling_closed": True,
            "zero_reward_range_branch_closed": True,
            "uncertified_SDP_solver_numerical_error_closed": False,
            "controlled_state_family_constructed": False,
            "C2_or_C3_promotion": False,
            "quantum_advantage": False,
            "publication_priority_exhaustive": False,
        },
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(RESULT.read_bytes()).hexdigest()
    print(
        f"status={STATUS}; AQ-SDP Bayes-Gram stability audit passed: "
        f"{len(fixed_measurement_rows)} fixed-measurement, "
        f"{len(exact_optimum_rows)} exact-optimum, {len(repair_rows)} repair, "
        f"{len(projection_repair_rows)} projection-repair, "
        f"{len(sampling_rows)} uniform-sampling, "
        f"{len(general_sampling_rows)} general-sampling, "
        f"{len(primal_dual_interval_rows)} primal-dual rows, "
        f"one zero-range row, result_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
