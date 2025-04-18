





k_l = [0.0026, 0.0052, 0.0021, 0.00642, 0.00642]
tau_l = [20.0000, 20.0000, 20.0000, 6.8250, 6.8250]
rho_l = [1.2000, 1.2000, 1.0000, 1.0000, 1.0000]
Cp_l = [3.6000, 3.4000, 3.0600, 3.7500, 3.7500]
Wb_l = [0.0000, 0.0005, 0.0005, 0.0005, 0.0005]
Cblood = [4.2, 4.2, 4.2, 4.2, 4.2]
cx = [2, 2, 2, 1, 1]
cy = [4, 4, 4, 2, 2]
cz = [1 ,1 ,1, 1, 1]

Skin_layer = 'Skin_1st'
type_ = []

Skin_l = Skin_layer
KEYS = ['Skin_1st','Skin_2nd','Skin_3rd','Tumor','Gold_Shell']

if Skin_layer=='Skin_3rd_blood': Skin_l='Skin_3rd'
if Skin_layer=='Tumor1' or Skin_layer=='Tumor2': Skin_l='Tumor'
    
index = KEYS.index(Skin_l)
index_L=5 if Skin_layer=='Tumor2' else 6 if Skin_layer=='Skin_3rd_blood' else index
if type_=='Same': index = 1


def k_l(): # Conduction coefficient (W/cm.deg. C)
    # W/cm. deg.C        
    return k_l[index]
        
def tau_l(): # Time delay (Second)
    # The time delay is assumed to be zero in 2006 paper
    # second
    return tau_l[index]
    
def rho_l():#  density
    # gr/(cm^3 )
    return rho_l[index]
        
def Cp_l(): # heat capacity 
    # J/(gr * deg.C)
    return Cp_l[index]
    
def Wb_l():    
    # gr/(cm^3*second)
    return Wb_l[index]
    
def Cblood(): 
    # J/(g*deg.C)
    return Cblood[index]   

def cxyz():
    if index_L==6: cy[2] = 2
    return cx[index] ,cy[index], cz[index]


def Len_():
       # LX = 1                       # Skin width (cm)
       LX = 1.00/2                    # Skin width (cm)
       Ly = 1.00
       L1 = [0   , 0.008, 0.208, 0.108, 0.158]
       L2 =[0.008, 0.208, 1.208, 0.308, 0.258]
       Lx1 = [0, 0, 0, 0.34, 0.4]
       Lx2 = [LX, LX, LX , LX, LX]
       Ly1 = [0, 0, 0, 0.34, 0.4]
       Ly2 = [Ly, Ly, Ly, 0.66, 0.6]
       return L1[index],L2[index],Lx1[index],\
                 Lx2[index],Ly1[index],Ly2[index]
    
def Len_t2(): # nonddimensionalize parameters
    # LX = 1                       # Skin width (cm)
    LX = 1.00/2                    # Skin width (cm)
    Ly = 1.00
    L1 = [0   , 0.008, 0.208, 0.058,0.058,0.058, 0.438 ]#0.158, 0.208, 0.438]
    L2 =[0.008, 0.208, 1.208, 0.258,0.258,0.258, 1.208 ]#0.258, 0.308, 1.208]
    Lx1 = [0,   0,  0, 0.34, 0.34, 0.34, 0 ]
    Lx2 = [LX, LX, LX, LX,    LX,  LX ,  LX]
    Ly1 = [0,  0,   0, 0.34, 0.34, 0.34, 0.23]
    Ly2 = [Ly, Ly, Ly, 0.66, 0.66, 0.66, 0.68]
    #Lz = L1
    return L1[index_L],L2[index_L],Lx1[index_L],\
                Lx2[index_L],Ly1[index_L],Ly2[index_L]