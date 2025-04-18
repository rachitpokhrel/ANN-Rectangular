

import tensorflow as tf
import Domain.Constants as c
import PINN.PINNUtilities as pu
import PINN.MergedModel as mm
        

def create(self,x0, ntime):
    if self.Time[0]>0: 
        input_0  = pu.add_ConstColumn(self.ft(x0), C = ntime, NCol = 3)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(input_0)
            T0 = mm(inputs = input_0) # T is a tuple with size of 3                
    
        T_grad = tape.gradient(T0, input_0)
        T0t = [T_grad[j][:,3:4] for j in range(len(T_grad))]
    
        u_0 = tuple(tf.reshape(T0[i], x0[i].shape[:-1]+(1,)) for i in range(len(T0)))
    
        ut_0 = tuple(tf.reshape(T0t[i], x0[i].shape[:-1]+(1,)) for i in range(len(T0t)))

    else:
        u_0 = tuple(self.T0['Skin']*tf.ones(xi.shape[:-1]+(1,),dtype=c.DTYPE) for xi in x0)
        ut_0 = tuple(tf.zeros(xi.shape[:-1]+(1,),dtype=c.DTYPE) for xi in x0)
    return  u_0, ut_0
  
    
    