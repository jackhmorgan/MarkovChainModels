"""This code was copied from qiskit_algorithms 0.3.0 (which is no longer supported) and modified to compatible
with qiskit > 1.0. The original code can be referenced at https://qiskit-community.github.io/qiskit-algorithms/_modules/qiskit_algorithms/amplitude_estimators/mlae.html#MaximumLikelihoodAmplitudeEstimation"""


from typing import List, Sequence
import numpy as np
from scipy.optimize import brute

from qiskit import ClassicalRegister
from qiskit.circuit import QuantumCircuit, QuantumRegister

from .EstimationProblem import EstimationProblem

def construct_mlae_circuits(
        estimation_problem: EstimationProblem, measurement: bool = False
    ) -> List[QuantumCircuit]:
        """Construct the Amplitude Estimation w/o QPE quantum circuits.

        Args:
            estimation_problem: The estimation problem for which to construct the QAE circuit.
            measurement: Boolean flag to indicate if measurement should be included in the circuits.

        Returns:
            A list with the QuantumCircuit objects for the algorithm.
        """
        # keep track of the Q-oracle queries
        circuits = []

        num_qubits = max(
            estimation_problem.state_preparation.num_qubits,
            estimation_problem.grover_operator.num_qubits,
        )
        q = QuantumRegister(num_qubits, "q")
        qc_0 = QuantumCircuit(q, name="qc_a")  # 0 applications of Q, only a single A operator

        # add classical register if needed
        if measurement:
            c = ClassicalRegister(len(estimation_problem.objective_qubits))
            qc_0.add_register(c)

        qc_0.compose(estimation_problem.state_preparation, inplace=True)

        for k in [0,1,2,4]:
            qc_k = qc_0.copy(name="qc_a_q_%s" % k)

            if k != 0:
                qc_k.compose(estimation_problem.grover_operator.power(k), inplace=True)

            if measurement:
                # real hardware can currently not handle operations after measurements,
                # which might happen if the circuit gets transpiled, hence we're adding
                # a safeguard-barrier
                qc_k.barrier()
                qc_k.measure(estimation_problem.objective_qubits, c[:])

            circuits += [qc_k]

        return circuits

nevals = max(10000, int(np.pi / 2 * 1000 * 2 * 4))

def _default_minimizer(objective_fn, bounds):
    return brute(objective_fn, bounds, Ns=nevals)[0]

def _get_counts(
    circuit_results: Sequence[dict[str, int]], estimation_problem: EstimationProblem
) -> tuple[list[int], list[int]]:
    """Get the good and total counts.

    Returns:
        A pair of two lists, ([1-counts per experiment], [shots per experiment]).

    Raises:
        AlgorithmError: If self.run() has not been called yet.
    """
    one_hits = []  # h_k: how often 1 has been measured, for a power Q^(m_k)
    all_hits = []
    for counts in circuit_results:
        all_hits.append(sum(counts.values()))
        one_hits.append(
            sum(
                count
                for bitstr, count in counts.items()
                if estimation_problem.is_good_state(bitstr)
            )
        )

    return one_hits, all_hits

def compute_mle(
        circuit_results: list[dict[str, int]],
        estimation_problem: EstimationProblem,
        return_counts: bool = False,
    ) -> float | tuple[float, list[int]]:
        """Compute the MLE via a grid-search.

        This is a stable approach if sufficient grid-points are used.

        Args:
            circuit_results: A list of circuit outcomes. Can be counts or statevectors.
            estimation_problem: The estimation problem containing the evaluation schedule and the
                number of likelihood function evaluations used to find the minimum.
            return_counts: If True, returns the good counts.
        Returns:
            The MLE for the provided result object.
        """
        good_counts, all_counts = _get_counts(circuit_results, estimation_problem)

        # search range
        eps = 1e-15  # to avoid invalid value in log
        search_range = [0 + eps, np.pi / 2 - eps]

        def loglikelihood(theta):
            # loglik contains the first `it` terms of the full loglikelihood
            loglik = 0
            for i, k in enumerate([0,1,2,4]):
                angle = (2 * k + 1) * theta
                loglik += np.log(np.sin(angle) ** 2) * good_counts[i]
                loglik += np.log(np.cos(angle) ** 2) * (all_counts[i] - good_counts[i])
            return -loglik

        est_theta: float = _default_minimizer(loglikelihood, [search_range])
        estimation = np.sin(est_theta) ** 2

        if return_counts:
            return estimation, good_counts
        
        return estimation