
import Domain.Constants as c
import PINN.AdamUtilities as au
import PINN.OptimizerMethod as om
import tensorflow as tf
import Domain.DomainUtilities as du

T0_new = au.T0_new
keys = au.keys
Nf , N_points = au.Nf , au.N_points
C_band3 = au.C_band3
layers = au.layers
minpoints = au.minpoints
LB, UB = au.LB, au.UB
Blood_Band = au.Blood_Band
Blood_Band_ND = au.Blood_Band_ND
Time = au.Time
input_args =  au.input_args
DTYPE = c.DTYPE
P0 = 0.75




def dTb_Tt():
    return 2/8

def SGD():
    return False

def optimizer():
    return "adamw"

def optimizer_Wt():
    return om.methodWeight()

def optimizer_method():
    return om.method(SGD(), optimizer())

def isCaseNew():
    return au.isCaseNew()

def segments():
    return au.segments()

def isBloodPresent():
    return segments[0]

def isTumorPresent():
    return segments[1]

def PDEw():
    return au.PDEw()

def activationFunction():
    return 'tanh'

def LRMethod():
    return 'cycle'

def argsTissue():
    return input_args['args_tissue']

def argsBlood():
    return input_args['args_blood']

def tissueName():
    return keys

def PRange():
    return du.PRange()

def dP(P_range):
    return P_range[1] - P_range[0]

def Pmin(PRange):
    return PRange[0]

def mDir():
    return [2, 0, 2, 2, 0, 2]

def D1Shape(hvd):
    return int(minpoints/hvd.size())  #Size of datasets in each batch

def d3():
    return layers['Skin'][0]

def ft(x, n = 0):
    if n==0: n=d3-1
    return tuple(tf.reshape(i,(-1,n)) for i in x)

def Tn(bloodBand, tn):
    time = bloodBand['time']
    Tn = time[1] - time[0] if tn == 0 else tn
    return Tn

def sigma():
    if isCaseNew():
        return 0.04
    else:
        return 0.01

def isdTSquare():
    if isCaseNew():
        return True
    else:
        return False

def isScattering():
    if isCaseNew():
        return True
    else:
        return False

def wallGroup():
    return [[0,1,2],[3],[4,5,6,7],[8],[9,10,11,12],[13,14,15],
            [16],[17,18,19,20],[21],[22,23,24,25]]


def lossNum_b():
    return 13 #+ 6  # 6 for sloperecovery

def lossNum_t():
    return 21 + 31  + 6 if isTumorPresent else 21

def lossNum_wall():
    return len(wallGroup())  # group walls together

w_t = [tf.Variable(1.0, dtype = c.DTYPE) for ii in range(lossNum_t())]
w_b = [tf.Variable(1.00, dtype = c.DTYPE) for ii in range(lossNum_b())]   # dWPINN
w_wall = [tf.Variable(1.00, dtype = c.DTYPE) for ii in range(lossNum_wall())]


def learningRateCycleStart(path, load_w_t):
    return 0 if len(path)==0 else 2 if load_w_t else 1

def learningRateCycle(path, load_w_t):
    lr_cycle = [(0.004, 8e-4),(8e-4, 1e-4),(2e-4, 6e-5),(5e-5, 2e-5),(8e-6, 9e-6)]
    return lr_cycle[learningRateCycleStart(path, load_w_t)]

def X_blood_sign():
    return (tf.TensorSpec([None,d3-2], DTYPE), tf.TensorSpec([None,d3-2], DTYPE),
            tf.TensorSpec([None,d3-2], DTYPE), tf.TensorSpec([None,d3-2], DTYPE),
            tf.TensorSpec([None,d3-2], DTYPE), tf.TensorSpec([None,d3-2], DTYPE))

def step0():
    return tf.Variable(0, trainable=False)

def fd(x):   
    P_shape =len(x[-1].shape)-1
    if c.P_Variable:
        x_out = tuple(tf.concat([xx, tf.random.uniform(xx.shape[:P_shape]+(1,),dtype = c.DTYPE)],\
        axis=P_shape) if len(xx)>0 else xx for xx in x)
    else:
        if d3==5:
            x_out = tuple(tf.concat([xx, P0 *tf.ones(xx.shape[:P_shape]+(1,),dtype = c.DTYPE)], 
            axis = P_shape) if len(xx)>0 else xx for xx in x)
        else: 
            x_out = x
    return x_out


def ft(x, n = 0):
    if n==0: n=d3-1
    return tuple(tf.reshape(i,(-1,n)) for i in x)

  
def add_ConstColumn(X, C = [1,1,1,1,1], NCol = 2):
    if not(isinstance(C,list)): C=[C]
    if isinstance(X,tuple):
        if len(C)!=len(X): C = [C[0] for _ in range(len(X))]
        return tuple(tf.concat((X[j][:,:NCol], C[j]* tf.ones((X[j].shape[0],1),
                                                             dtype=c.DTYPE), X[j][:,NCol:]), axis=1) for j in range(len(X)))
    else:
        if isinstance(C, list) or isinstance(C, tuple):
            return tuple(tf.concat((X[:,:NCol], C[j]* tf.ones((X.shape[0],1),
                                                              dtype=c.DTYPE), X[:,NCol:]), axis=1) for j in range(len(C)))  
        else:
            return tf.concat((X[:,:NCol], C* tf.ones((X.shape[0],1),
                                                    dtype=c.DTYPE), X[:,NCol:]), axis=1)

