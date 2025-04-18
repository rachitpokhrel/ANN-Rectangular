
import tensorflow as tf
import Domain.Constants as c
import PINN.PINNUtilities as pu
import PINN.BloodProperties as bp
import PINN.Net as net
import PINN.Q as q
import Domain.NDDomains as ndds
        

def IC(x0, T00, T0t =[], plt_loss=0): 
    #  X0 is 4D --> 0  should be added to column 3
    # IC  : U0 , dU0 
    U0_pred, U0t_pred = net.skinMerged(pu.add_ConstColumn(x0, C=0, NCol=3), mixed=3)
    
    if pu.isdTSquare: 
        if plt_loss:
            Loss_0_data = [tf.square(U0_pred[ii]-T00[ii])+tf.square(U0t_pred[ii] - T0t[ii])
                        for ii in range(len(U0_pred))]
        else:  
            Loss_0_data = [tf.square(U0_pred[ii]-T00[ii]) for ii in range(len(U0_pred))]
            Loss_0_data += [tf.square(U0t_pred[ii] - T0t[ii]) for ii in range(len(U0t_pred))]
    else:
        Loss_0_data = [tf.square(U0_pred[ii]-T00[ii]) for ii in range(len(U0_pred))]
        
    if plt_loss: return Loss_0_data
    else:       #  MSE0 of key 
        Loss_0 = [tf.reduce_mean(Loss_0_data[ii]) for ii in range(len(Loss_0_data))]
        return  Loss_0 

def BCXY(X , Bound ='lb', col = 0, plt_loss = 0): 
    #  X is 4D --> C_band[key][0,col]  should be added to column col 
    Loss_bc, LossT_IFC, LossdT_IFC = [] ,[] ,[]
    C_band = pu.LB if Bound =='lb' else pu.UB     
    if plt_loss==0:
        XD = pu.add_ConstColumn(X, C=[C_band[ky][0,col] for ky in pu.keys] ,NCol = col)
    else: 
        XD = tuple(pu.add_ConstColumn(xi, C= C_band[pu.keys[ii]][0,col] ,NCol = col)[0]\
                if len(xi)>0 else tf.ones([2,pu.d3], dtype=pu.DTYPE) for ii,xi in enumerate(X))

    u_pred, *ud_pred = net.skinMerged(XD, mixed = 0)
    
    if pu.isTumorPresent():  # if pu.isTumorPresent()
            u_neghbor = transformation_IFC_Xy(XD, col = col) # {key:(u, gradu) ,...}
    else: u_neghbor = {}
    #----------------------------------------------------------------------#
    for ii , key in enumerate(pu.keys):
        if key in u_neghbor.keys():  # IFC Boundary conditions
            Ci = ndds.ndds(key).params.k_l()/ndds.ndds(key).characteristicLengths()[col],
            Cj = ndds.ndds(u_neghbor[key][0]).params.k_l()/ndds.ndds(u_neghbor[key][0]).characteristicLengths()[col]
            if plt_loss==0:        
                LossT_IFC.append(tf.reduce_mean(tf.square(u_pred[ii]-u_neghbor[key][1])))
            
                LossdT_IFC.append(tf.reduce_mean(tf.square(Ci*ud_pred[col][ii]-Cj*u_neghbor[key][2]))) 
            else:
                LossT_IFC.append(tf.square(Ci*ud_pred[col][ii]-Cj*u_neghbor[key][2])+
                                        tf.square(u_pred[ii]-u_neghbor[key][1]))
        #----------------------------------------------        
        elif plt_loss==0:  # insulation conditions
            Loss_bc.append(tf.reduce_mean(tf.square(ud_pred[col][ii])))
        
    if plt_loss==0: return Loss_bc, LossT_IFC, LossdT_IFC
    else: return LossT_IFC


def BCZ(z_lb, z_ub, plt_loss=0):  # Z is 4D (x,y,z,t,p) 

    Zlb =pu.add_ConstColumn(z_lb, C=[pu.LB[ky][0,2] for ky in pu.keys] ,NCol = 2)
    
    Zub = tuple(pu.add_ConstColumn(zi, C= pu.UB[pu.keys[ii]][0,2] ,NCol = 2)[0]\
                        if len(zi)>0 else zi for ii,zi in enumerate(z_ub))
    
    # to make the code runs faster, lb and ub are combined first and 
    # apu.fter estimation they will separted vs their neighbors
    Ci = [ndds.ndds(key).params.k_l()/ndds.ndds(key).characteristicLengths()[2] for key in pu.keys]
    i3, i2 = pu.keys.index('Skin_3rd'), pu.keys.index('Skin_2nd')
    if pu.isTumorPresent():
        it2, it1 = pu.keys.index('Tumor2'), pu.keys.index('Tumor1')
        ig = pu.keys.index('Gold_Shell')
    # function estimation for LB
    U_lb,_, _, Ud_lb, Ulbt = net.skinMerged(Zlb, mixed = 0) 
    # function estimation for UB
    Zub_tissue = transform_fun([Zlb[i2], Zlb[i3]],['Skin_2nd','Skin_3rd'], 
                                                        ['Skin_1st','Skin_2nd'])
    Zub_new = (Zub_tissue[0],)
    for ii in range(len(Zub)-1):
        if pu.keys[ii+1]=='Tumor1': Zub_new +=(Zlb[it2],)
        elif pu.keys[ii+1]=='Skin_2nd': Zub_new +=(Zub_tissue[1],)
        else: Zub_new +=(Zub[ii+1],)
        
    U_ub,_, _, Ud_ub,_ = net.skinMerged(Zub_new, mixed = 0) 
    #---------------------------------------------------------------------#     
    U_IFC = (((U_ub[0],Ci[0]*Ud_ub[0]) ,(U_lb[i2],Ci[i2]*Ud_lb[i2])),  #'Skin_2nd & 1st
            ((U_ub[i2],Ci[i2]*Ud_ub[i2]) ,(U_lb[i3],Ci[i3]*Ud_lb[i3]))) #'Skin_2nd& 3rd
    
    if pu.isTumorPresent():
        # function estimation for neighbors
        Transform_from = ['Tumor1','Tumor2','Gold_Shell','Gold_Shell']
        Transform_to = ['Skin_2nd','Skin_3rd','Tumor1','Tumor2']
        Z_neighbor = (Zlb[it1], Zub[it2], Zlb[ig], Zub[ig])
        Z_neighbors = transform_fun(Z_neighbor,Transform_from, Transform_to)
        U_n,_, _, Ud_n,_ = net.skinMerged(Z_neighbors, mixed = 0)
        #-------------------------------------------------------------------#
        U_IFC += (((U_n[i2],Ci[i2]*Ud_n[i2]),(U_lb[it1],Ci[it1]*Ud_lb[it1])),  #'Tumor1'-Skin_2nd
                    ((U_ub[it1],Ci[it1]*Ud_ub[it1]),(U_lb[it2],Ci[it2]*Ud_lb[it2])),  #'Tumor2,'Tumor1
                    ((U_n[it1],Ci[it1]*Ud_n[it1]) ,(U_lb[ig],Ci[ig]*Ud_lb[ig])),  #'Tumor1'-Gold
                    ((U_n[i3],Ci[i3]*Ud_n[i3]) ,(U_ub[it2],Ci[it2]*Ud_ub[it2])),   #'Tumor2'-Skin_3rd
                    ((U_n[it2],Ci[it2]*Ud_n[it2]) ,(U_ub[ig],Ci[ig]*Ud_ub[ig])))  #'Tumor2'-Gold                                    
    #---------------------------Loss_estimations-----------------------------# 
    LossT_IFC, LossdT_IFC, Lossz =[], [], []   
            
    if pu.isdTSquare:  #  Consider delay BC for air
        if plt_loss==0:
            Lossz0 = [40*tf.reduce_mean(tf.square(Ud_lb[0]-ndds.ndds('Skin_1st')))].Bi1*(U_lb[0]- ndds.T_inf() + ndds.ndds('Skin_1st').Rt*Ulbt[0])
        else: LossT_IFC.append(tf.square(Ud_lb[0]-ndds.ndds('Skin_1st').Bi1*(
                                U_lb[0]- ndds.T_inf() + ndds.ndds('Skin_1st').Rt*Ulbt[0])))
    else:
        if plt_loss==0:Lossz0 = [60*tf.reduce_mean(tf.square(Ud_lb[0]-ndds.ndds('Skin_1st').Bi1*(U_lb[0]- ndds.T_inf())))]
        else: LossT_IFC.append(tf.square(Ud_lb[0]-ndds.ndds('Skin_1st').Bi1*(U_lb[0]- ndds.T_inf())))
    #-----------------------------------------------------------------------# 
    w = [10,8,10,4,7,6,6,6] if pu.isdTSquare else [7,7,4,1]
    for j,i in enumerate(range(len(U_IFC))):  #--> 8 (7) for case with blood --> 16(14)
        if plt_loss==0:
            LossT_IFC.append(w[j]*tf.reduce_mean(tf.square(U_IFC[i][0][0]-U_IFC[i][1][0])))
            LossdT_IFC.append(w[j]*tf.reduce_mean(tf.square(U_IFC[i][0][1]-U_IFC[i][1][1])))
        else: 
            LossT_IFC.append(tf.square(U_IFC[i][0][0]-U_IFC[i][1][0])+\
                            tf.square(U_IFC[i][0][1]-U_IFC[i][1][1]))
    #-----------------------------------------------------------------------#       
    if plt_loss==0: 
        Lossz.append(tf.reduce_mean(tf.square(Ud_ub[i3])))     
    # total loss has loss0 (1) + IFC (total 8) + UBz skin_3rd (max=2 or (1))           
    if plt_loss==0:return Lossz0 + LossT_IFC + LossdT_IFC + Lossz #(6)
    else: return LossT_IFC  # Loss of LBZ + UBZ (the last 2 losses for pu.UBumor2, ub_gold)

def transform_fun(X,Transform_from =[], Transform_to =[]):
    # Transform_to should not contain identical members
    # the length of X should be the same as pu.keys
    # X shoul be created for Transform-from
    Same_Domain = ['Tumor1','Tumor2','Gold_Shell']
    temp = tf.ones([3,pu.d3], dtype=c.DTYPE)
    Xnew = ()
    #-----------------------------------------------------------------------#
    if len(Transform_to)>0:
        for ii, key_frwd in enumerate(pu.keys):
            if key_frwd in Transform_to:
                jj = Transform_to.index(key_frwd)
                key_inv = Transform_from[jj]
                if key_frwd==key_inv or (key_frwd in Same_Domain and key_inv in Same_Domain):
                    Xnew +=(X[jj],)
                else:
                    Xnew += (ndds.ndds(key_frwd).forward(ndds.ndds(key_inv).inverse(
                            X[jj],t0 =pu.Time[0]),t0 = pu.Time[0]),)
            else:Xnew +=(temp,)
    return Xnew
#---------------------------------------------------------------------------#    
def transformation_IFC_Xy(X, col = 0):  # X is 5D (x,y,z,t,p) --> for BC on X, col=0, etc.
    # inputs X transform into their neighbors for IFC
    Transform_from = ['Tumor1','Tumor2','Gold_Shell','Gold_Shell']
    Transform_to = ['Skin_2nd','Skin_3rd','Tumor1','Tumor2']

    x_from = tuple(X[pu.keys.index(ii)] for ii in Transform_from)
    Xtransform = transform_fun(x_from, Transform_from, Transform_to)
    
    U,*Ugrads = net.skinMerged(Xtransform, mixed = 0)

    U_neighbor = {Transform_from[ii]: (kk, U[pu.keys.index(kk)],
                    Ugrads[col][pu.keys.index(kk)]) for ii,kk in enumerate(Transform_to[:2])}
    #---------------------------------------------------------------#
    ## concat the gold-Shells between Tumor1 and Tumor2
    g_ind, t1_ind = pu.keys.index('Gold_Shell'),pu.keys.index('Tumor1'),
    t2_ind = pu.keys.index('Tumor2')
    ug = tf.where(X[g_ind][:,2:3]<0.5, U[t1_ind], U[t2_ind])
    ug_grad = tf.where(X[g_ind][:,2:3]<0.5, Ugrads[col][t1_ind], Ugrads[col][t2_ind])
    U_neighbor['Gold_Shell'] = ('Tumor1', ug, ug_grad)
                                    
    return U_neighbor  


def PDE(x  ,plt_loss = 0): 
    # mixed = 1  # For PDE in net_skinMerged function
    # x = (x_f for: 'Skin_3rd' ,'Skin_2nd','Skin_1st') : tuple  
    # Is_local_w : True | False :  If local weights applied
    Loss_f , Loss_f_data = [] ,[]
    T, Tt, Txx, Tyy, Tzz, Ttt = net.skinMerged(x, mixed = 1) 
    wf = pu.PDEw() if pu.isTumorPresent() else [20, 20, 10]#wf = [2,2,6,8]

    #Loss_pos = [tf.reduce_mean(100*tf.square(tf.where(-ut>=0,ut,0))) for ut in Tt] 

    for j,key in enumerate(pu.keys):
        ND = ndds.ndds(key)
        Tavr = T_avr_output_of_Blood(x[j])
        Q = q.QEstimation(x[j], ND, key)
        # physics informed
        if pu.isdTSquare:
            f_u = (ND.A*ND.Rt+1)*Tt[j] + ND.Rt*Ttt[j] - ND.F0[0]*Txx[j] - ND.F0[1]*Tyy[j] -\
                                ND.F0[2]*Tzz[j] - ND.A*(Tavr-T[j]) - Q/ND.Q_Coeff()
        else:
            f_u = Tt[j] -ND.F0[0]*Txx[j]- ND.F0[1]*Tyy[j]- ND.F0[2]*Tzz[j]-\
                                ND.A*(Tavr-T[j])- Q/ND.Q_Coeff()                              
        # 1 - MSE PDEs for 3 skin_layers
        Loss_f_data.append(tf.square(f_u))
        if plt_loss==0:
            Loss_f.append(wf[j]*tf.reduce_mean(Loss_f_data[j]))
            
    if plt_loss: return Loss_f_data
    else: return Loss_f#+Loss_pos    # it is a list

def T_avr_output_of_Blood(x_f = []):
    # Estimating u_out at the exit of Arterial blood vessels 
    if pu.isBloodPresent():
        # I need 6 inputs here, I just need the 3rd one -->[2]
        x_out = tf.concat([bp.out_b[2] * tf.ones([x_f.shape[0],1],dtype=c.DTYPE),x_f[:,3:]],1)
        temp = tf.ones([4,pu.d3-2], dtype=c.DTYPE)
        Xin = (temp, temp, x_out, temp, temp, temp)
        Tout,_ = net.bloodMerged(Xin)
        Tavr = Tout[2] * pu.dTb_Tt   
    else:
        Tavr = ndds.T_in()
    return Tavr

def Tissues(X_dist_batch): 
    # Nt = len(pu.keys) 
    #if len(X_dist_batch)==8:
    x0, x_f, x_lbx, x_ubx, x_lb_uby , x_lbz, x_ubz, T0,T0t = X_dist_batch
    #if len(X_dist_batch)==9:
    #    x0, x_f, x_lbx, x_ubx, x_lb_uby , x_lbz, x_ubz, T0,xmore = X_dist_batch
    #    x_f = tuple(tf.concat([x_f[i],xmore[i]],axis=1) for i in range(len(xmore)))
    #------------------------------------------------------------------------#     
    T00 = tuple(tf.reshape(i,(-1,1)) for i in T0)
    T0t = tuple(tf.reshape(i,(-1,1)) for i in T0t)
    
    Loss_f = PDE(pu.ft(x_f,pu.d3)) #--> Nt losses
    # Losses on BC x-y-z-dir, IC, PDE, and IFC 
    Loss_0 = IC(pu.ft(x0), T00, T0t) #--> Nt losses

    Loss_xub = symmetry_Xub(pu.ft(x_ubx))  # -> len(Tissues) losses
    #------------------------#
    # Loss_x for ['Skin-1st - Skin-3rd] -->3 (3)
    # LossT_x and LossdT_x for IFC [tumor1-2nd, tumor2-3rd, gold-tumor] --> 6
    Loss_xlb, LossT_xlb, LossdT_xlb = BCXY(pu.ft(x_lbx), 'lb', col = 0)

    Y = pu.ft(x_lb_uby)
    # Loss_y for ['Skin-1st - Skin-3rd ]--> 3
    # LossT_y and LossdT_y for IFC [tumor1-2nd, tumor2-3rd, gold-tumor] --> (6) 
    Loss_ylb, LossT_ylb, LossdT_ylb = BCXY(Y, 'lb', col = 1)
    Loss_yub, LossT_yub, LossdT_yub = BCXY(Y, 'ub', col = 1)
    Loss_y = [Loss_ylb[i] + Loss_yub[i] for i in range(len(Loss_yub))]
    LossT_y = [LossT_ylb[i] + LossT_yub[i] for i in range(len(LossT_ylb))]
    LossdT_y = [LossdT_ylb[i] + LossdT_yub[i] for i in range(len(LossdT_ylb))]
    #------------------------#
    # loss0 (1) + IFC (2)|(3)|(7)|(8) + UBz skin_3rd (2 with blood) or (1))-->(4)|(6)|(9)|(11)
    Loss_z = BCZ(pu.ft(x_lbz), pu.ft(x_ubz))
    
    #----------# Loss_f + Loss_0 + Symmetery + LossX_Y insulation -> 3*Nt + 6
    # Nt(Lf) + Nt(L0)+ Nt(L_ubx) + 3[L_lbx] + 3[Ly] + 1 [L_z0]
    Loss_T = Loss_f + Loss_0 + Loss_xub + Loss_xlb[:3] + Loss_y[:3]#   + Loss_yub[:3]    # Do the same for tumor ????
    if pu.isTumorPresent(): Loss_T += LossT_xlb[:3] + LossdT_xlb[:3] + LossT_y[:3] + LossdT_y[:3] # 12   
        
    return Loss_T + Loss_z

#Symmetry Boundary conditions on X --> X_ub  ( All pu.keys members)
def symmetry_Xub(x_ub, plt_loss = 0): 
    #  Xub is 4D -->cx should be added to column 0
    _, ux_ubx, _,_,_ = net.skinMerged(pu.add_ConstColumn(x_ub,
                [pu.UB[ky][0,0] for ky in pu.keys] ,NCol=0), mixed = 0)
        
    if plt_loss==0: return [tf.reduce_mean(tf.square(ux)) for ux in ux_ubx]             
    else: return [tf.square(ux) for ux in ux_ubx] 