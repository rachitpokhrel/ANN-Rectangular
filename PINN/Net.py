

import Domain.Constants as c
import PINN.MergedModel as mm
import PINN.PINNUtilities as pu
import tensorflow as tf
        
       

def skinMerged(self,X, mixed = tf.constant(0), T_num = 2):  
    # T_num : the number of tissues in which X is defined when len(X)!=len(self.Tissues_name)
    if len(X)!=len(self.Tissues_name):
        temp = tf.ones([3,self.d3], dtype=c.DTYPE)
        X = tuple(X[0] if i==T_num else temp for i in range(len(self.Tissues_name)))
    if mixed==2: # For blood ends
        T = mm(inputs = X)
        return T[T_num]
    else: return self.skinMerged_models(X, mixed = mixed)

@tf.function
def skinMerged_models(self, X , mixed = 0):#tf.constant(0)):
    # mixed = 1 for PDE    # mixed = 3 for IC
    # mixed = 0 for BCs   # mixed = 2 for Power
    # inputs and outputs are tuples: X is a tuple
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            T = mm(inputs = X) # T is a tuple with size of 3                
        # inputs is also a list of size of 3
        if mixed < 3 or pu.isdTSquare():
            T_grad = tape1.gradient(T, X)
            Tx = [T_grad[j][:,0:1] for j in range(len(T_grad))]
            Ty = [T_grad[j][:,1:2] for j in range(len(T_grad))]
            Tz = [T_grad[j][:,2:3] for j in range(len(T_grad))]
            Tt = [T_grad[j][:,3:4] for j in range(len(T_grad))]
        
    if mixed == 1:  # for PDE
        T2_gradx = tape2.gradient(Tx, X)
        Txx = [T2_gradx[j][:,0:1] for j in range(len(T2_gradx))]
        T2_grady = tape2.gradient(Ty, X)
        Tyy = [T2_grady[j][:,1:2] for j in range(len(T2_grady))]
        T2_gradz = tape2.gradient(Tz, X)
        Tzz = [T2_gradz[j][:,2:3] for j in range(len(T2_gradz))]
        if pu.isdTSquare():
            T2_gradt = tape2.gradient(Tt, X)
            Ttt = [T2_gradt[j][:,3:4] for j in range(len(T2_gradt))]
        del tape2, tape1
        if pu.isdTSquare(): return T,Tt,Txx,Tyy,Tzz,Ttt 
        else: return T,Tt,Txx,Tyy,Tzz,()
    
    elif mixed==0: return T,Tx,Ty,Tz,Tt # otherwise mixed == 0
    else: 
        if pu.isdTSquare(): return T, Tt  # for IC mixed = 3
        else: return T, ()


@tf.function
def bloodMerged(self, X):#, mixed = tf.constant(0)):
    # inputs and outputs are tuples
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        # we have 3 inputs [lenght , time, Power] and one output [T]
        T = mm(inputs = X) # u_ is a list with size of 6
    T_grad = tape.gradient(T, X)
    T_d = [T_grad[j][:,0:1] for j in range(len(X))]
    return T, T_d