
#import PINN.AdamUtilities as au

def PRange():
    return [0.5, 1.0]

def dP(P_range):
    return P_range[1] - P_range[0]

def Pmin(PRange):
    return PRange[0]

def isBloodPresent():
    return True