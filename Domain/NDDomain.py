import tensorflow as tf
import numpy as np
import Domain.Constants as c
import Domain.Parameters as p
import Domain.DomainUtilities as du
        


class NDDomain:

    def __init__(self, Skin_layer = 'Skin_1st', tn = 1, remove_dim =[]):
        self.Skin_l = Skin_layer
        self.tn = tn
        self.remove_dim = remove_dim
        if Skin_layer in ['Arterial', 'Venous']:
            self.Skin_l='Skin_3rd'
        p.Skin_layer = self.Skin_l
        self.cx , self.cy, self.cz = p.cxyz()
        self.Bi1 = c.h_inf*self.characteristicLengths()[2]/p.k_l()
        self.T_alpha = p.k_l()/(p.rho_l()*p.Cp_l())
        self.A = self.ts()*p.Cblood()*p.Wb_l()/(p.rho_l()*p.Cp_l())
        self.F0 = tuple(self.T_alpha*self.ts()/(L**2) for L in self.characteristicLengths())
        self.Rt = p.taul()/self.ts()

        self.params = p


    def ts(self): 
        return self.tn

    def characteristicLengths(self):
        # if X = (x-Lx)/Lsx, 
        # Y = (Y-Ly)/Lsy, 
        # Z= (z-Lz)/Lsz with X, Y , and Z dimensionless parameters, 
        # and Ls = Li being charactersitic lenght                        
        return p.Len_t2()[3]-p.Len_t2()[2]/self.cx, p.Len_t2()[5]-p.Len_t2()[4]/self.cy, p.Len_t2()[1]-p.Len_t2()[0]/self.cz






    def characteristicTemperature(self):
        Tr = c.T0
        Ts = Tr - c.deltaT_Tissue 
        if du.isBloodPresent:
            Ts = Tr - c.deltaT_Blood
        return Tr, Ts
        
    def ndT(self, Temperature):
        Tr, Ts = self.characteristicTemperature()
        theta = (Temperature-Tr)/(Tr-Ts)
        return theta



    def Q_Coefficient(self):
        # if Qhat = Q*Qs     be a dimensionless Q 
        # then Qs is 
        Tr, Ts = self.characteristicTemperature()
        Qs = p.rho_l*p.Cp_l/self.ts()*(Tr-Ts)
        return Qs


    def Bi1(self):
        return c.h_inf*self.characteristicLengths()[2]/p.k_l()

    def T_alpha(self):
        return p.k_l()/(p.rho_l()*p.Cp_l())

    def A(self):
        return self.ts()*p.Cblood*p.Wb_l()/(p.rho_l()*p.Cp_l())

    def F0(self):
        return tuple(self.T_alpha()*self.ts()/(L**2) for L in self.characteristicLengths())

    def Rt(self):
        return p.tau_l()/self.ts()

    def L0(self, cell_num):
        Label = ['z','x','z','z','x','z']
        Lz1, Lz2, Lx1, Lx2, _, _ = p.Len_t2()#TODO: Calculations
        # L0 = [L_entry,L_out]
        if Label[cell_num]=='z': 
            L0 = [Lz1,(Lz2-Lz1)/self.cz]
        else:                    
            L0 = [Lx1,(Lx2-Lx1)/self.cx]
        return L0

    def forward(self, var, dimension, t0=0):
        # convert to nondmensional variables
        cd = self.matchCondition(var)

        Lsx, Lsy, Lsz = self.characteristicLengths()
        Tr, Ts = self.characteristicTemperature()
        Lz, _, Lx, _ , Ly, _ = p.Len_t2()  #consider this

        coeffx = [Lsx, Lsy, Lsz, self.ts(), du.dP(du.PRange())]
        coeff0 = [Lx, Ly, Lz, t0, du.Pmin(du.PRange())]

        for i in self.remove_dim: 
                    coeffx.pop(i)
                    coeff0.pop(i)

        if cd > 1 and cd < 10:
            if tf.is_tensor(var):
                Coeff_X = tf.constant([coeffx],dtype = tf.float32)
                Coeff_0 = tf.constant([coeff0],dtype = tf.float32)
                var = (var - Coeff_0)/ Coeff_X
            else:
                Coeff_X = np.array([coeffx])
                Coeff_0 = np.array([coeff0])
                var = (var - Coeff_0)/ Coeff_X
                var = var.astype(np.float32)        

        elif cd == 1:
            if dimension == 'T': 
                var = (var-Tr)/(Tr-Ts)
            elif dimension == 'z': 
                var =(var-Lz)/Lsz
            elif dimension =='time': 
                var = (var-t0)/self.ts()
            elif dimension=='x': 
                var =(var-Lx)/Lsx
            elif dimension=='y': 
                var =(var-Ly)/Lsy
                
        elif cd == 10:
            var = {0:(var[0]-Lx)/Lsx, 1:(var[1]-Ly)/Lsy, 2:(var[2]-Lz)/Lsz, 3:(var[3]-t0)/self.ts()}
                    
        return var

    #Blood vessels
    def forwardBlood(self, L0, t0=0, dimension='L'):
        # convert data in blood vessels to nondmensional variables
        # We just have 2D variables [x(or y or z), t]
        # Temperature is similar to the other tissues, so:
        # ND.Forward_Var(Var,d='T')
        # L0 =[Lentry, Lout]
        Coeff_X = np.array([L0[1],self.ts()])

        cd = self.matchCondition(var)

        if cd > 1 and cd < 10:
            var = (var-np.array([L0[0],t0]))/ Coeff_X.flatten()[None,:]
        elif cd == 1:
            if dimension == 'L':
                var =(var-L0[0])/Coeff_X[0]
            elif dimension == 'time':
                var = (var-t0)/self.ts()
        elif cd == 10:
            var ={0:(var[0]-L0[0])/Coeff_X[0], 1:(var[1]-t0)/self.ts()}              
        return var

    def Bi_blood_wall2(self, endx=False):
        Lsx, Lsy, Lsz = self.characteristicLengths()
        Bi = 95.23
        Bi = [73.81, 92.26, 122.86,73.81, 92.26, 122.86]

        Dim_wall =((0,1,1),(1,1,2,2,0),(0,0,1,1))
        wall_label = ((Lsx,Lsy,-Lsy),(Lsy,-Lsy,Lsz,-Lsz,Lsx),(Lsx,-Lsx,Lsy,-Lsy))
        Band =()
        if endx:
            Dim_wall=((0,1,1,2),(1,1,2,2,0),(0,0,1,1))
            wall_label= ((Lsx,Lsy,-Lsy,Lsz),(Lsy,-Lsy,Lsz,-Lsz,Lsx),(Lsx,-Lsx,Lsy,-Lsy))
            Band = (('lb','lb','ub','lb'),('lb','ub','lb','ub','lb'),('lb','ub','lb','ub'))

        Dim_wall +=Dim_wall
        Band +=Band

        wall_label +=wall_label   
        return Dim_wall , tuple(tuple(Bi[i]*wall_label[i][j] for j in range(len(wall_label[i]))) for i in range(len(wall_label))), Band

    def inverse(self, var, dimension, t0=0):
        # return to our real geometry
            cd = self.matchCondition(var)

            Lsx,Lsy,Lsz = self.characteristicLengths()
            Tr, Ts = self.characteristicTemperature()
            Lz,_,Lx,_,Ly,_ = p.Len_t2()#consider this
            
            coeffx = [Lsx, Lsy, Lsz, self.ts(), du.dP(du.PRange())]
            coeff0 = [Lx, Ly, Lz, t0, du.Pmin(du.PRange())]

            for i in self.remove_dim: 
                    coeffx.pop(i)
                    coeff0.pop(i)

            if (cd > 1 and cd < 10) :
                if tf.is_tensor(var):
                    Coeff_X = tf.constant([coeffx],dtype = tf.float32)
                    Coeff_0 = tf.constant([coeff0],dtype = tf.float32)
                    var = var* Coeff_X + Coeff_0
                else:
                    Coeff_X = np.array([coeffx])
                    Coeff_0 = np.array([coeff0])
                    var = var* Coeff_X + Coeff_0
                    var = var.astype(np.float32)
                
            elif cd == 1:
                if dimension =='T':  
                    var = var*(Tr-Ts)+Tr
                elif dimension =='z': 
                    var = var*Lsz + Lz
                elif dimension =='time': 
                    var = var*self.ts()+t0
                elif dimension =='x': 
                    var = var*Lsx+Lx 
                elif dimension =='y': 
                    var = var*Lsy+Ly
            
            elif cd == 10:
                var ={0:var[0]*Lsx+Lx, 1:var[1]*Lsy+Ly, 2:var[2]*Lsz+Lz, 3:var[3]*self.ts()+ t0}

            return var

    #Helper Function
    def matchCondition(variable):
        if isinstance(variable,(float,int)):
            return 1
        elif isinstance(variable,dict): 
            return 10
        elif tf.is_tensor(variable): 
            return variable.shape[1]
        elif variable.shape[0] == variable.size:
            return 1 
        else: 
            return variable.shape[-1]