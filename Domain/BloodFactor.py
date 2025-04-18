import numpy as np

# Blood vessels properties
def factor_blood(self,cell_num,m_level=3):
    # m is a number between [0,6]
    if m_level ==3:
        m, sign_ = self.cell_num_3levels(cell_num)
    # alpha: Heat transfer Coefficient between blood and tissue
    alpha = 0.2  # W/cm^2.C
    # CB: Heat capacity of blood
    CB = 4.134   # J/cm^3.C
    # vm: Velocity of blood flow 
    NLb,NWb,Lb,vm =self.Vessel_dim()
    # Pm: Vessel perimeter (cm)
    Pm = 2*(NWb[m]+NLb[m])
    if m==0:
        Pm=2*NWb[m]+NLb[m]
    # Fm: Area of Cross section
    Fm = NWb[m]*NLb[m]
    # Mm : Mass flow of blood vessels
    # Mm = vm[m]*Fm
    factorm = alpha*Pm/(CB*vm[m]*Fm) #* Lb[m]       
    # pdot: decreased blood flow rate
    Pdot = 0.5e-3 # 1/s
    g_m = Pdot/vm[m]#*Lb[m]
    return sign_*factorm, sign_*g_m          
"""
    Vessel coordinates for system
    7 layers of blood vessels
"""
def Vessel_dim(self):
    NLb,NWb,Lb,vm=np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3)     
    NLb[0], NWb[0], Lb[0], vm[0] = 0.1, 0.1, 0.4, 8  # cm , cm, cm, cm/s
    M1 = vm[0]*NWb[0]*NLb[0]           # correct M1: not symmetry
    
    for i in range(1,3):
        NLb[i] = 2**(-1/3)*NLb[i-1]
        NWb[i] = 2**(-1/3)*NWb[i-1]
        Lb[i] = 2**(-1/2)*Lb[i-1]
        vm[i] = M1/(2*NWb[i]*NLb[i])
    NWb[0] /=2    
    return NLb,NWb,Lb,vm
#-------------------------------------------------------------------------#
def cell_num_3levels(self, cell_num): # in symmetry
    # cell_num : between 0-9 : 0-4 for Arteries | 5-9 for veins
    m_level_i = 0 if cell_num in [0,3] else 1 if cell_num in [1,4] else 2
    sign_ = 1 if cell_num in [3,4,5] else -1
    return m_level_i, sign_
#-------------------------------------------------------------------------#