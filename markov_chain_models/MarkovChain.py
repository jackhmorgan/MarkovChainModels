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

class MarkovChain(QuantumCircuit):
    
    def __init__(self, time_steps, prob_gb, prob_bg):
        
        theta_naught = 2*np.arccos(np.sqrt((prob_bg)/(prob_gb+prob_bg)))
        theta_0 = 2*np.arccos(np.sqrt(1-prob_gb))
        theta_1 = 2*np.arccos(np.sqrt(prob_bg))
        
        n = time_steps+1
        
        circ = QuantumCircuit(n)
        
        circ.ry(theta_naught, [0],'theta_naught')
        
        for i in range(n-1):
            circ.ry(theta_0, [i+1], 'theta_0')
            circ.cry(theta_1 - theta_0, [i], [i+1], 'theta_1')
            
        super().__init__(circ.num_qubits,name='MC')
        self.append(circ.to_gate(),range(circ.num_qubits)) 
