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

from .MarkovChain import MarkovChain

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit

from qiskit.circuit.library import QFT, LinearAmplitudeFunction
from typing import Optional
import numpy as np


class DerivativePricing(QuantumCircuit):   
            
        def _AdderBaseQFT(self, value):
            '''Adds the constant value to the price register in the QFT basis'''
            circ_a = QuantumCircuit(self.num_size)
            
            for i in range(self.num_size):
                lam = value * (2**self.fractional_precision) * np.pi / (2**(i))
                circ_a.p(lam, i)
            
            return circ_a.to_gate(label='Add Value')
        
        def _MCBinTree(self):
    
            dt = self.time_tot/self.time_steps
            
            sigma_off = [0.2,0.3]
            r = [0.2,0.1]
        
        
            circ_b = QuantumCircuit(1+(2*self.time_steps)+self.num_size)
    
            for i in range(self.time_steps):
            
                sigma = sigma_off[0]+min(max(1.2*i*dt,0),0.1)
                mu = r[0] - ((sigma**2)/2)

                lu = (mu*dt)+(sigma*np.sqrt(dt))
                ld = (mu*dt)-(sigma*np.sqrt(dt))

                lucirc = self._AdderBaseQFT(lu).control(num_ctrl_qubits=2, ctrl_state='00')
                ldcirc = self._AdderBaseQFT(ld).control(num_ctrl_qubits=2, ctrl_state='10')

                circ_b.append(lucirc, [i+1,self.time_steps+1+i]+list(range(-self.num_size,0,1)))
                circ_b.append(ldcirc, [i+1,self.time_steps+1+i]+list(range(-self.num_size,0,1)))

                sigma = sigma_off[1]+min(max(1.2*i*dt,0),0.1)
                mu = r[1] - ((sigma**2)/2)

                lu = (mu*dt)+(sigma*np.sqrt(dt))
                ld = (mu*dt)-(sigma*np.sqrt(dt))

                lucirc = self._AdderBaseQFT(lu).control(num_ctrl_qubits=2, ctrl_state='01')
                ldcirc = self._AdderBaseQFT(ld).control(num_ctrl_qubits=2, ctrl_state='11')

                circ_b.append(lucirc, [i+1,self.time_steps+1+i]+list(range(-self.num_size,0,1)))
                circ_b.append(ldcirc, [i+1,self.time_steps+1+i]+list(range(-self.num_size,0,1)))
            return circ_b.to_gate(label='Price Evolution')
        
        def _Payoff(self):

            f_max = ((2**(self.integer_precision+1))-(2.0**(-self.fractional_precision)))
            po_max = f_max - self.strike_price
            
            circ_p = LinearAmplitudeFunction(
                num_state_qubits=self.integer_precision+self.fractional_precision+1,
                slope=[0,1],
                offset=[0,0],
                domain=(0, f_max),
                image=(0, po_max),
                breakpoints=[0,self.strike_price],
                rescaling_factor=self.c_approx,
            )
            return circ_p
        
        def __init__(self,
            strike_price: float,
            time_steps: int,
            prob_gb: Optional[float] = 0.1,
            prob_bg: Optional[float] = 0.3,
            integer_precision: Optional[int] = 1,
            fractional_precision: Optional[int] = 6,
            starting_price: Optional[float] = 1.0,
            time_tot: Optional[float] = 1/12,
            r: Optional[float] = 0.1,
            c_approx: Optional[float] = 0.05,
            name: Optional[str] = 'DP'
            ) -> None:               
                
        
            self.strike_price = strike_price
            self.time_steps = time_steps
            self.integer_precision = integer_precision
            self.fractional_precision = fractional_precision
            self.num_size = integer_precision+fractional_precision+1
            self.starting_price = starting_price
            self.time_tot = time_tot
            self.r = r
            self.c_approx = c_approx
            self.f_max = ((2**self.integer_precision)-(2.0**(-self.fractional_precision)))
            
            payoff = self._Payoff()
            qubits = 1+(2*time_steps)+payoff.num_qubits
            
            self.post_processing = payoff.post_processing
            self.objective = qubits-payoff.num_ancillas-1
            
            circ = QuantumCircuit(qubits)
            
            circ.append(MarkovChain(time_steps,prob_gb,prob_bg).to_gate(),range(time_steps+1)) #prepare Markov Chain
            
            circ.h(range(1+time_steps,1+(2*time_steps))) #prepare binomial tree

            circ.append(QFT(self.num_size,0,do_swaps=False, inverse=False, insert_barriers=False).to_gate(), 
                        range(1+(2*time_steps),1+(2*time_steps)+self.num_size)) #switch price register to base QFT

            '''Note: We calculate exolution of the price in log space,
            then convert at the end using e^x approx 1+x'''
            circ.append(self._AdderBaseQFT(np.log(starting_price)),
                        range(1+(2*time_steps),1+(2*time_steps)+self.num_size)) 

            circ.append(self._MCBinTree(),
                        range(1+(2*time_steps)+self.num_size))

            circ.append(self._AdderBaseQFT(1),
                        range(1+(2*time_steps),1+(2*time_steps)+self.num_size)) #converts to normal space
            
            circ.append(QFT(self.num_size,0,do_swaps=False, inverse=True, insert_barriers=False).to_gate(), 
                        range(1+(2*time_steps),1+(2*time_steps)+self.num_size)) #converts to computational basis 

            circ.append(payoff.to_gate(),
                        list(range(1+(2*time_steps),qubits))) #peicewise function = price - strike price if price > strike price
         
            super().__init__(circ.num_qubits, name=name)
            self.append(circ.to_gate(),self.qubits)
