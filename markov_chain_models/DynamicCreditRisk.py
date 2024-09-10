'''
Copyright 2024 Jack Morgan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

from qiskit import QuantumCircuit
import numpy as np
import math
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.circuit.library.arithmetic import PolynomialPauliRotations, WeightedAdder
from qiskit.circuit.library import QFT
from typing import Optional
from qiskit.circuit.library.arithmetic import IntegerComparator

from .MarkovChain import MarkovChain

class DynamicCreditRisk(QuantumCircuit):
    def _AdderBaseQFT(self, value):
        circ_a = QuantumCircuit(self.num_sum_qubits)
        for i in range(self.num_sum_qubits):
            lam = value * np.pi * 2**(self.fractional_precision - i)
            circ_a.p(lam, i)   
        return circ_a.to_gate(label='Add_value')
    
    def _OneStepGrowths(self):
        circ_g = QuantumCircuit(self.num_sum_qubits+1)
        for ctrl in [0,1]:
            if self.growth_possibilities[ctrl] != 0:
                add = self._AdderBaseQFT(self.growth_possibilities[ctrl])
                circ_g.append(add.control(ctrl_state=ctrl), circ_g.qubits)
        return circ_g.to_gate(label='Add_growth')
    
    def __init__(self,
                 loss: int,
                 time_steps: int,
                 prob_gb: Optional[float] = 0.009708737864077669,
                 prob_bg: Optional[float] = 0.1111111111111111,
                 growth_possibilities: Optional[list] = [0.771,0],
                 fractional_precision:Optional[int] = 2):
        
        self.num_sum_qubits = 2+fractional_precision+math.ceil(np.log2(growth_possibilities[0]*time_steps))
        self.fractional_precision = fractional_precision
        self.time_steps = time_steps #not including the steady state solution
        self.growth_possibilities = growth_possibilities
        non_zeros = [increment for increment in growth_possibilities if increment != 0]
        
        smallest_increment = min(non_zeros, key=abs)
        self.scaled_loss = loss+(2**(-fractional_precision-1)*smallest_increment)
        
                 
        M = MarkovChain(time_steps, prob_gb, prob_bg).to_gate()
        
        iQ = QFT(self.num_sum_qubits,0,do_swaps=False, inverse=True, insert_barriers=False).to_gate()
        
        C = self._AdderBaseQFT(-self.scaled_loss)
        
        circ = QuantumCircuit(M.num_qubits+self.num_sum_qubits)
        self.objective = circ.num_qubits-1  #qubit to measure and/or objective in QAE
        
        circ.append(M, qargs=range(time_steps+1)) #prepare Markov Chain Qubits
        
        #circ.append(Q, qargs=list(range(M.num_qubits,M.num_qubits+self.num_sum_qubits)))
        circ.h(list(range(M.num_qubits,M.num_qubits+self.num_sum_qubits)))
        
        for i in range(self.time_steps):
            i += 1
            circ.append(self._OneStepGrowths(), [i]+list(range(M.num_qubits,M.num_qubits+self.num_sum_qubits)))
        
        circ.append(C, qargs=list(range(M.num_qubits,M.num_qubits+self.num_sum_qubits)))
        circ.append(iQ, qargs=list(range(M.num_qubits,M.num_qubits+self.num_sum_qubits))) 
        #circ.append(C.to_gate(), qargs=list(range(M.num_qubits,M.num_qubits+C.num_qubits))) #Compare the sum of the losses to our input value
        #circ.h(list(range(M.num_qubits,M.num_qubits+self.num_sum_qubits)))
        
        super().__init__(circ.num_qubits, name = str(loss)+'_loss_'+str(time_steps)+'_steps')
        self.append(circ.to_gate(),range(circ.num_qubits))       
