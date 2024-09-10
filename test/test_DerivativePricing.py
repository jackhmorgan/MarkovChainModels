import unittest
from ddt import ddt, data, unpack
import numpy as np

from markov_chain_models import DerivativePricing
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT
from qiskit import QuantumCircuit

@ddt
class TestDerivativePricing(unittest.TestCase):
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
            
            max = np.argmax(probabilities)

            np.testing.assert_array_almost_equal(expected, 
                                                 probabilities,
                                                 decimal=2)
        
    def setUp(self):
         self.dp = DerivativePricing(strike_price=1,
            time_steps=3,
            prob_gb = 0.1,
            prob_bg = 0.3,
            integer_precision = 1,
            fractional_precision = 6,
            starting_price = 1.0,
            time_tot = 1/12,
            r = 0.1,
            c_approx = 0.05,
            name = 'DP'
            )

    @data(
         ([0.539, 0.4615], 1.0, 3, 0.07, 0.11, 4, 1),
         ([0.539, 0.4615], 0.95, 3, 0.07, 0.11, 4, 1),
         ([0.539, 0.4615], 1.0, 4, 0.07, 0.11, 4, 1),
         ([0.539, 0.4615], 1.0, 3, 0.1, 0.3, 4, 1),
         ([0.539, 0.4615], 1.0, 3, 0.1, 0.3, 5, 1),
         ([0.539, 0.4615], 1.0, 3, 0.1, 0.3, 4, 2),
    )
    @unpack
    def test_DP_circuit(self,
                        expected,
                        strike_price,
                        time_steps,
                        prob_gb,
                        prob_bg,
                        integer_precision,
                        fractional_precision,
                        ):
         circuit = DerivativePricing(strike_price,
                            time_steps,
                            prob_gb,
                            prob_bg,
                            integer_precision,
                            fractional_precision,
                            )
         self.AssertExpectedCiruitResult(expected=expected, 
                                         circuit=circuit,
                                         qubits = [circuit.objective])
    @data(
         {'value': 0.5},
         {'value': 0.25}
    )
    @unpack     
    def test_AdderBaseQFT(self,
                        value,
                        ):
         adder = self.dp._AdderBaseQFT(value=value)
         fractional_precision = self.dp.fractional_precision
         expected_index = value*2**fractional_precision
         expected = np.zeros(2**(adder.num_qubits))
         expected[int(expected_index)] = 1
         iqft = QFT(adder.num_qubits, inverse=True, do_swaps=False)


         circ = QuantumCircuit(adder.num_qubits)
         circ.h(circ.qubits)
         circ.compose(adder, circ.qubits, inplace=True)
         circ.compose(iqft, circ.qubits, inplace=True)
         
         self.AssertExpectedCiruitResult(expected=expected,
                                         circuit=circ
                                         )
         

         
        
         
if __name__ == '__main__':
    unittest.main()
            