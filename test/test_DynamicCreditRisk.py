# This Python script is a unit test for a class called `DynamicCreditRisk` which is part of a module
# named `markov_chain_models`. Here's a breakdown of what the script does:
import unittest
from ddt import ddt, data, unpack
import numpy as np

from markov_chain_models import DynamicCreditRisk
from qiskit.quantum_info import Statevector

@ddt
class TestDynamicCreditRisk(unittest.TestCase):
    """Test Dynamic Credit Risk Circuit."""
    # variety of test cases with expected outcomes predetermined
    @data(
         ([0.91995611, 0.08004389], 1, 3, 0.009708737864077669, 0.1111111111111111, [0.771,0], 2),
         ([0.93104713, 0.06895287], 1, 4, 0.009708737864077669, 0.1111111111111111, [0.771,0], 2),
         ([0.77161287, 0.22838713], 1, 3, 0.1, 0.3, [0.771,0], 2),
         ([0.93359595, 0.06640405], 1, 3, 0.009708737864077669, 0.1111111111111111, [1.2,0], 2),
         ([0.91690231, 0.08309769], 1, 3, 0.009708737864077669, 0.1111111111111111, [0.771,0], 4),
    )
    @unpack
    def test_DCR_circuit(self,
                         expected, 
                        loss,
                        time_steps,
                        prob_gb,
                        prob_bg,
                        growth_possibilities,
                        fractional_precision,
                        ):
         # create dcr class
         circuit = DynamicCreditRisk(loss,
                 time_steps,
                 prob_gb,
                 prob_bg,
                 growth_possibilities,
                 fractional_precision)
         
         # determine measurement probabilities of objective qubit
         statevector = Statevector(circuit)
         probabilities = statevector.probabilities([circuit.objective])

         # assert that they match the expected probabilities
         np.testing.assert_array_almost_equal(expected, 
                                                 probabilities,
                                                 decimal=3)
         
if __name__ == '__main__':
     unittest.main()