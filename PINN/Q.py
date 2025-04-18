
import Domain.Constants as c
import PINN.PINNUtilities as pu
import PINN.QLaser as ql
import tensorflow as tf
import numpy as np
       
        

def QEstimation(self, x, ND, Skin_layer=[]):
    alpha , Reff = 1.8, 0.1     
    Gauusian_xy = tf.exp((tf.square(ND.Inverse_Var(x[:,0:1],d='x')-0.5)+
                    tf.square(ND.Inverse_Var(x[:,1:2],d='y')-0.5))/\
                        (-2*self.sigma**2))/tf.sqrt(2*np.pi*pu.sigma**2)

    if pu.isScattering(): 
        Q = ql.Qz(x[:,2:3], Skin_layer)  
            
    else: Q = alpha *tf.exp(-alpha*ND.Inverse_Var(x[:,2:3],d='z'))
    
    if c.P_Variable:
        return Q*(x[:,4:5]*pu.dP() + pu.Pmin())*(1-Reff)*Gauusian_xy  #
    else:
        return Q*c.P0*(1-Reff)*Gauusian_xy  # Iin = P0 *(1 - Reff) 