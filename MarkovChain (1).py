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