import PINN.LossSkin as ls
import PINN.LossWall as lw
import Domain.Constants as c
import PINN.PINNUtilities as pu
import PINN.MCMH as MCMH
import PINN.T0 as T0
import tensorflow as tf
        
        
        

def main(self, Xt): 
    x0, x_f, x_lbx, x_ubx, x_lby , x_lbz, x_ubz, T0, T0t = Xt
    
    x_f = self.pde(x_f)
    x0, T0, T0t = self.ic(x0,T0, T0t)
    x_ubx = self.ubX(x_ubx)
    x_lbz, x_ubz = self.z_IFC( x_lbz, x_ubz)      
    if pu.isTumorPresent():
        x_lbx, x_lby = self.xy_IFC(x_lbx, x_lby)
    
    return (x0, x_f, x_lbx, x_ubx, x_lby , x_lbz, x_ubz, T0, T0t)

def ft_inv(self, x, n = 0, D =0):
    D1 = D if D else pu.D1Shape() 
    n1 = n if n else pu.d3()-1 
    return tuple(tf.reshape(i,(D1,-1, n1)) for i in x)

def concat_wall(self, xwall):
    xwall = tuple(tuple(tf.reshape(xwall[j][i],(-1,pu.d3())) for i in range(len(xwall[j]))) for j in range(len(xwall)))
    return tuple(tf.concat(xwi ,axis =0) for xwi in xwall)


def wall(self, Xb): # MCMH for Xwall
    xend, xwall = Xb
    D_shape = xwall[0][0].shape[0]
    Loss_wall_data,*_, D = lw.wall(xwall, plt_loss=1)
    #--------------Create proposed xwall and Estimate their loss--------------#
    if len(xwall[0][0].shape)==2:
        Nw = [[xwall[i][j].shape[0] for j in range(len(xwall[i]))] for i in range(len(xwall))]
    elif len(xwall[0][0].shape)==3:
        Nw = [[xwall[i][j].shape[0]*xwall[i][j].shape[1] for j in range(len(xwall[i]))] 
                                for i in range(len(xwall))]

    xwall_prop = MCMH.MCMH().new_prop_Wallpoints_tf(self.Wall_band['lb'], 
                                                self.Wall_band['ub'],Nw, D_shape) 

    xwall_prop = tuple(pu.fd(xwall_i) for xwall_i in xwall_prop)

    Loss_wall_prop,*_ = lw.wall(xwall_prop, plt_loss=1)
    #------------------------find alpha and Xwall_new-------------------------#
    alphaw = [Loss_wall_prop[i]/Loss_wall_data[i] for i in range(len(Loss_wall_data))]
    del Loss_wall_data, Loss_wall_prop
    Xwallnew = self.loss(alphaw ,self.concat_wall(xwall), self.concat_wall(xwall_prop))

    Xwall_new = tuple(tuple(tf.reshape(Xwallnew[j][D[j][ii]:D[j][ii+1],:], (D_shape,-1,pu.d3())) 
                    for ii in range(len(D[j])-1)) for j in range(len(xwall)))
    return (xend, Xwall_new)  # update new dataset

def pde(self, Xf):
    Xf = pu.ft(Xf,pu.d3())
    Nf = [xii.shape[0] for xii in Xf]
    Xf_prop = MCMH.MCMH().new_prop_points_tf(pu.LB, pu.UB, Nf,
                                        pu.C_band3, Key_order = pu.keys)
    Xf_prop = pu.fd(Xf_prop)
    #-----------------------------------------------------------------------#
    #Xf_more = mcmh_fun.new_prop_points_tf(pu.LB, pu.UB, [max(int(xii.shape[0]*1e-3),2) for xii in Xf],
    #                                      pu.C_band3, Key_order = pu.keys)
    #Xf_more = pu.fd(Xf_more)
    #Xf_more = tuple(Xf_more[i] if More_[i] and Nf[i]<9e5 else  tf.constant([]) for i in range(len(Xf_more)))
    #-----------------------------------------------------------------------#
    Loss_f_data = ls.PDE(Xf, plt_loss = 1)
    Loss_f_prop = ls.PDE(Xf_prop ,plt_loss = 1)
    alphaf = [Loss_f_prop[i]/Loss_f_data[i] for i in range(len(Loss_f_data))]
    del Loss_f_data, Loss_f_prop
    return pu.ft_inv(self.loss(alphaf ,Xf, Xf_prop),pu.d3())

def ic(self, X0, T00, T00t, ntime):
    X0, T00, T00t = pu.ft(X0),pu.ft(T00,1), pu.ft(T00t,1)
    Loss_0_data = ls.IC(X0, T00 , T00t, plt_loss = 1)
    N0 = [xii.shape[0] for xii in X0]

    X0_prop=MCMH.MCMH().new_prop_points_tf(pu.LB,pu.UB, N0,
                                pu.C_band3, Bound = 4,Key_order = pu.keys)
    X0_prop = pu.fd(X0_prop)
    T0p, T0t_p = T0.create(X0_prop, ntime)
    #-----------------------------------------------------------------------#
    #X0_more=mcmh_fun.new_prop_points_tf(pu.LB,pu.UB,[max(int(xii.shape[0]*1e-3),2)for xii in X0],
    #                          pu.C_band3, Bound = 4,Key_order = pu.keys)
    #X0_more = pu.fd(X0_more)
    #X0_more = tuple(X0_more[i] if More_[i] and N0[i]<2e5 else  tf.constant([]) for i in range(len(X0_more)))
    #-----------------------------------------------------------------------#
    Loss_0_prop = ls.IC(X0_prop, T0p,T0t_p, plt_loss = 1)
    alpha0 = [Loss_0_prop[i]/Loss_0_data[i] for i in range(len(Loss_0_data))]
    del Loss_0_data, Loss_0_prop
    X0new = pu.ft_inv(self.loss(alpha0 ,X0, X0_prop))
    #X0new = self.Metropolis_Hasting_Alg_loss(alpha0 ,X0, X0_prop,X0_more)
    T0_new, T0t_new = T0.create(X0new, ntime)
    return X0new, T0_new, T0t_new

def ubX(self, Xub_x):
    Xub_x = pu.ft(Xub_x)
    Loss_ubx_data = ls.symmetry_Xub(Xub_x, plt_loss = 1)
    
    Xub_x_prop=MCMH.MCMH().new_prop_points_tf(pu.LB,pu.UB,[xi.shape[0] for xi in Xub_x]
                    ,pu.C_band3, Bound = 1, Dir = 'ub', Key_order = pu.keys)
    Xub_x_prop = pu.fd(Xub_x_prop)
    
    Loss_ubx_prop = ls.symmetry_Xub(Xub_x_prop, plt_loss = 1)
    
    alphax = [Loss_ubx_prop[i]/Loss_ubx_data[i] for i in range(len(Loss_ubx_data))]
    del Loss_ubx_prop,Loss_ubx_data

    return pu.ft_inv(self.loss(alphax ,Xub_x, Xub_x_prop))

def z_IFC(self, x_lbz, x_ubz):
    Xlbz, Xubz = pu.ft(x_lbz), pu.ft(x_ubz)     
    Nz = [xii.shape[0] for xii in Xlbz]
    Xlbz_prop = MCMH.MCMH().new_prop_points_tf(pu.LB, pu.UB, Nz,
                                pu.C_band3, Bound = 3, Dir = 'lb',Key_order = pu.keys)
    Xlbz_prop = pu.fd(Xlbz_prop)
    if pu.isTumorPresent():
            #Nt = [Xubz[i].shape[0] if i>=3 else 0 for i in range(len(Xubz))]
        Nt = [Xubz[i].shape[0] for i in range(len(Xubz))]
        Xubz_prop = MCMH.MCMH().new_prop_points_tf(pu.LB,pu.UB,Nt,
                        pu.C_band3, Bound = 3, Dir = 'ub',Key_order = pu.keys)
        Xubz_prop = pu.fd(Xubz_prop)
    else:Xubz_prop = Xubz
    # lossz = [lb of all tissues] if Is_tumor : [ub: tumor2, and Gold]
    Loss_z_data= ls.BCZ(Xlbz, Xubz, plt_loss = 1) 
    Loss_z_prop= ls.BCZ(Xlbz_prop, Xubz_prop, plt_loss = 1)
    
    alphaz = [Loss_z_prop[i]/Loss_z_data[i] for i in range(len(Loss_z_data))]
    #----------------------------------------------------------------------#
    #Xlbz_more=mcmh_fun.new_prop_points_tf(pu.LB,pu.UB,[max(int(xii.shape[0]*1e-3),2) for xii in Xlbz],
    #                          pu.C_band3, Bound = 3, Dir = 'lb',Key_order = pu.keys)
    #Xlbz_more = pu.fd(Xlbz_more)
    #Xlbz_more = tuple(Xlbz_more[i] if More_[i] and Nz[i]<2.2e5 else  tf.constant([]) for i in range(len(Xlbz_more)))
    #-----------------------------------------------------------------------#
    Xlbznew = self.loss(alphaz[:len(Xlbz)], Xlbz, Xlbz_prop)#, Xlbz_more)
    
    if pu.isTumorPresent():
        alphaub = [0, 0, 0, 0, alphaz[-2], alphaz[-1]]
        Xubznew = self.loss(alphaub, Xubz, Xubz_prop)
    else: Xubznew = Xubz
            
    return pu.ft_inv(Xlbznew), pu.ft_inv(Xubznew)
    #except: return Xlbznew, pu.ft_inv(Xubznew)

def XY_IFC(self, x, y): # in case we have either tumor or blood or both       
    Col =[0,1] if pu.isTumorPresent() else [1] # if is_blood but not tumor
    X_new =() if pu.isTumorPresent() else (x,) # if is_blood but not tumor
    for col in Col : # x and y      
        X = pu.ft(x) if col==0 else pu.ft(y)
        # for tumor1,tumor2, Gold
        d = [3,4,5] 
        Nt = [X[i].shape[0] if i in d else 0 for i in range(len(X))]
    
        X_prop = MCMH.MCMH().new_prop_points_tf(pu.LB, pu.UB, Nt, pu.C_band3,
                                Bound = col+1, Dir = 'lb', Key_order = pu.keys) 
        X_prop = pu.fd(X_prop)
        # loss is defined for elements in d
        Loss_X_data = ls.BCXY(X,      'lb', col = col, plt_loss=1)
        Loss_X_prop = ls.BCXY(X_prop, 'lb', col = col, plt_loss=1)
        if col==1:
            Loss_y_data = ls.BCXY(X,      'ub', col = col, plt_loss=1)
            Loss_y_prop = ls.BCXY(X_prop, 'ub', col = col, plt_loss=1)
            Loss_X_data = [Loss_y_data[i] + Loss_X_data[i] for i in range(len(Loss_X_data))]
            Loss_X_prop = [Loss_y_prop[i] + Loss_X_prop[i] for i in range(len(Loss_X_prop))]
        #--------------------------------------------------------------------#
        alphax = [0 for i in range(len(X))]
        for i,di in enumerate(d):alphax[di] = Loss_X_prop[i]/Loss_X_data[i] 
        X_new += (pu.ft_inv(self.loss(alphax, X, X_prop)),)
        
    return X_new

def loss(self,alpha, X, Xp, Xmore = []):
    N = [alpha[i].shape[0] if tf.is_tensor(alpha[i]) else 0  for i in range(len(alpha))]    
    C = tuple(tf.random.uniform([N[i],1], minval=0.01, maxval=1.0,dtype = c.DTYPE) 
                for i in range(len(N)))
    if len(Xmore): 
        return tuple(tf.concat([tf.where(alpha[i]>=C[i], Xp[i], X[i]),Xmore[i]],axis=0) 
                    if N[i]>0 else X[i] for i in range(len(N)))
    
    else: 
        return tuple(tf.where(alpha[i]>=C[i], Xp[i], X[i]) if N[i]>0 else X[i]
                                                        for i in range(len(N))) 