

import PINN.PINNUtilities as pu


def blood_name():
    return [i for i in range(len(pu.argsBlood()[1]))]

def blood_model():
    return {}   # Blood vessels and model

def wall_band():
    return {'ub': pu.Blood_Band['ub'],'lb': pu.Blood_Band['lb']}

def mid_point():
    return (pu.Blood_Band['lb'][2,0] + pu.Blood_Band['ub'][2,0])/2

def in_b():
    return [pu.Blood_Band['ub'][i ,pu.mDir()[i]] for i in range(3)]+\
                            [pu.Blood_Band['lb'][i ,pu.mDir()[i]] for i in range(3,6)]

def out_b():
    return [pu.Blood_Band['lb'][i,pu.mDir()[i]] for i in range(3)]+\
                            [pu.Blood_Band['ub'][i ,pu.mDir()[i]] for i in range(3,6)]


blood_name = blood_name()
blood_model = blood_model()    
wall_band = wall_band()
mid_point = mid_point()
in_b = in_b()
out_b = out_b()
out_b[1] , in_b[4] = mid_point, mid_point