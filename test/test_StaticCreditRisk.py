import unittest
from ddt import ddt, data, unpack
import numpy as np

from markov_chain_models import StaticCreditRisk
from qiskit.quantum_info import Statevector

@ddt
class TestStaticCreditRisk(unittest.TestCase):
    def AssertExpectedCiruitResult(self,
                            expected,
                            circuit,
                            qubits=None,
                            ):
            
            statevector = Statevector(circuit)
            if not qubits == None:  
                probabilities = statevector.probabilities(qubits)

            else:
                 probabilities = statevector.probabilities()

            np.testing.assert_array_almost_equal(expected, 
                                                 probabilities,
                                                 decimal=3)

    @data(
         ([0.28998934, 0.71001066], 1, 3, 0.1, 0.3, [[0.1,0.2],[0.15,0.25]], [[0.1,0.05],[0.15,0.1]], [1,2], 3),
         ([0.06236322, 0.93763678], 2, 3, 0.1, 0.3, [[0.1,0.2],[0.15,0.25]], [[0.1,0.05],[0.15,0.1]], [1,2], 3),
         ([0.28998934, 0.71001066], 1, 4, 0.1, 0.3, [[0.1,0.2],[0.15,0.25]], [[0.1,0.05],[0.15,0.1]], [1,2], 3),
         ([0.28998934, 0.71001066], 1, 3, 0.07, 0.11, [[0.1,0.2],[0.15,0.25]], [[0.1,0.05],[0.15,0.1]], [1,2], 3),
         ([0.39416232, 0.60583768], 1, 3, 0.1, 0.3, [[0.4,0.5],[0.10,0.20]], [[0.1,0.05],[0.15,0.1]], [1,2], 3),
         ([0.04833608, 0.95166392], 1, 3, 0.1, 0.3, [[0.1,0.2],[0.15,0.25]], [[0.5,0.65],[0.10,0.2]], [1,2], 3),
         ([0.40967211, 0.59032789], 1, 3, 0.1, 0.3, [[0.1,0.2],[0.15,0.25]], [[0.1,0.05],[0.15,0.1]], [2,2], 3),
         ([0.1166377, 0.8833623], 1, 3, 0.1, 0.3, [[0.1,0.2],[0.15,0.25]], [[0.1,0.05],[0.15,0.1]], [1,2], 4)
    )
    @unpack
    def test_SCR_circuit(self,
                         expected, 
                        loss,
                        time_steps,
                        prob_gb,
                        prob_bg,
                        default_probs,
                        sensitivities,
                        weights,
                        z_qubits,
                        ):
         
         circuit = StaticCreditRisk(loss,
                        time_steps,
                        prob_gb,
                        prob_bg,
                        default_probs,
                        sensitivities,
                        weights,
                        z_qubits,
                        )
         self.AssertExpectedCiruitResult(expected=expected, 
                                         circuit=circuit,
                                         qubits = [circuit.objective])
         
if __name__ == '__main__':
     unittest.main()