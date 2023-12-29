# Importing standard Qiskit libraries
from MarkovChain import MarkovChain

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# qiskit-ibmq-provider has been deprecated.
# Please see the Migration Guides in https://ibm.biz/provider_migration_guide for more detail.
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Estimator, Session, Options

# Loading your IBM Quantum account(s)
service = QiskitRuntimeService(channel="ibm_quantum")

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import XGate, RYGate, ZGate
from qiskit.circuit.library.arithmetic import PolynomialPauliRotations, WeightedAdder
from qiskit.circuit.library import QFT
from qiskit_finance.circuit.library import NormalDistribution
from qiskit.circuit.library.arithmetic import PolynomialPauliRotations
from qiskit import transpile
from scipy.stats import norm
import matplotlib.pyplot as plt
from qiskit.circuit.library import GroverOperator, IntegerComparator
from scipy.stats import linregress as lin
from json import JSONEncoder
import time, math, csv, json
from typing import Optional
from qiskit.algorithms import EstimationProblem
from scipy.stats import norminvgauss
from qiskit.circuit.library import QFT, LinearAmplitudeFunction
from qiskit import ClassicalRegister, execute
from typing import Optional
from scipy.stats import norm, linregress
from qiskit.circuit.library.arithmetic import PolynomialPauliRotations, WeightedAdder, IntegerComparator
from qiskit_finance.circuit.library import NormalDistribution
from qiskit_ibm_runtime.options import ExecutionOptions

class StaticCreditRisk(QuantumCircuit):
    
    def _OneStepUncertainty(self, default_probs, sensitivities):
        '''Note: default probs is a variable in the model that determines the probability of a loan defaulting
        This function returns a circuit that adds the probability of loan x defaulting in a single time step
        via a RY Pauli Polynomial'''
        
        '''Classically computes a linear approximation of the probability of default given the model 
        parameters determined by the state of the economy'''
        a_list = []
        b_list = []
        for i in range(self.groups):
            x_axis = []
            y_axis = []
            for j in range(2**self.z_qubits):
                pk = norm.cdf((default_probs[i] - (np.sqrt(sensitivities[i])*j))/(np.sqrt(1-sensitivities[i])))
                theta = 2*np.arcsin(np.sqrt(pk))
                x_axis.append(j)
                y_axis.append(theta/self.time_steps)
            approx = linregress(x_axis,y_axis)
            a_list.append(approx.slope)
            b_list.append(approx.intercept)
        
        '''Circuit that applies default probability to each qubit that represents a group of loans'''
        circ_u = QuantumCircuit(self.z_qubits+self.groups)
        
        for i in range(self.groups):
            poly = PolynomialPauliRotations(self.z_qubits,coeffs=[b_list[i], a_list[i]], basis='Y').to_gate()
            circ_u.append(poly,list(range(self.z_qubits))+[self.z_qubits+i])
            
        return circ_u.to_gate()
    
    def _MCUncertainty(self):
        '''Circuit that controlls the appropriate "one step Uncertainty" circuit for the good and bad economy 
        for each step in the Markov Chain'''
        circ_mcu = QuantumCircuit(self.time_steps+self.z_qubits+self.groups)
        
        Uncert_good = self._OneStepUncertainty(self.default_probs[0], self.sensitivities[0])
        Uncert_bad = self._OneStepUncertainty(self.default_probs[1], self.sensitivities[1])
        
        for i in range(self.time_steps):
            circ_mcu.append(Uncert_good.control(ctrl_state='0'), [i]+list(range(self.time_steps,circ_mcu.num_qubits)))
            
            circ_mcu.append(Uncert_good.control(ctrl_state='1'), [i]+list(range(self.time_steps,circ_mcu.num_qubits)))
            
        return circ_mcu.to_gate()
    
    
    def __init__(self, 
                 loss: int,
                 time_steps: int,
                 prob_gb: Optional[float] = 0.1,
                 prob_bg: Optional[float] = 0.3,
                 default_probs: Optional[list[list]] = [[0.1,0.2],[0.15,0.25]],
                 sensitivities: Optional[list[list]] = [[0.1,0.05],[0.15,0.1]],
                 weights: Optional[list] = [1, 2],
                 z_qubits: Optional[int] = 3,
                ) -> None :
         # circuit
        
        self.time_steps = time_steps #not including the steady state solution
        self.groups = len(default_probs[0]) #number of loans Y
        self.z_qubits = z_qubits #number of qubits used to represent our random variable
        self.default_probs = default_probs #model parameters
        self.sensitivities = sensitivities #model parameters
        
        
        N = NormalDistribution(z_qubits, mu=((2**z_qubits)-1)/2, sigma=((2**z_qubits)-1)/4, bounds=(0,(2**z_qubits)-1))
        M = MarkovChain(time_steps, prob_gb, prob_bg).to_gate()
        U = self._MCUncertainty()
        S = WeightedAdder(self.groups, weights) #manually adjust weights here
        C = IntegerComparator(S.num_sum_qubits, loss+1, geq=False)
        
        self.objective = -C.num_ancillas-1 #qubit to measure and/or objective in QAE
          
        circ = QuantumCircuit(1+time_steps+z_qubits+self.groups+S.num_ancillas+C.num_qubits)
        
        circ.append(N.to_gate(), qargs=range(1+time_steps,1+time_steps+z_qubits)) #prepare our random variable in a gaussian probability distribution
        circ.append(M, qargs=range(time_steps+1)) #prepare Markov Chain Qubits
        circ.append(U, qargs=range(1,1+time_steps+z_qubits+self.groups)) #encode the probability of a loan defaulting to the |1> state
        circ.append(S.to_gate(), qargs=range(1+time_steps+z_qubits,1+time_steps+z_qubits+S.num_qubits)) #add the loss from each group y if the loan's qubit is |1>
        circ.append(C.to_gate(), qargs=list(range(1+time_steps+z_qubits+self.groups, 1+time_steps+z_qubits+self.groups+S.num_sum_qubits))+list(range(-C.num_ancillas-1,0))) #Compare the sum of the losses to our input value
        
        super().__init__(circ.num_qubits)
        self.append(circ.to_gate(),range(circ.num_qubits))           