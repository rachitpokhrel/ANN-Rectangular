
import PINN.PINNUtilities as pu
import tensorflow as tf
import Net as net
import LossBlood as lb
import Domain.Constants as c
import Domain.NDDomains as ndds


def wall(xwall , plt_loss=0): #--> 26 losses for walls
    # blood vessels are located in the "Skin-3rd"
    uw_skin, duw_skin_Bi, xb_f, D = U_Wall_Skin(xwall)  # is a tuple
    uw_blood, duw_blood = net.bloodMerged(xb_f)
    
    # dT/dn *1/Bi + (Tw-Tb) = 0 --> Bi = Bi*L for lb and -Bi*L for ub
    if plt_loss:
        loss_f = lb.PDE(uw_skin, uw_blood, duw_blood, plt_loss=1)
        loss_walls = [loss_f[j] + tf.square(duw_skin_Bi[j]-
                        (uw_blood[j]*pu.dTb_Tt - uw_skin[j])) for j in range(len(uw_blood))]
    #-----------------------------------------------------------------------#
    else:
        loss_walls_ = [tf.reduce_mean(tf.square(duw_skin_Bi[j][D[j][ii]:D[j][ii+1],:]-
                    (uw_blood[j][D[j][ii]:D[j][ii+1],:]*pu.dTb_Tt - uw_skin[j][D[j][ii]:D[j][ii+1],:]))) 
                        for j in range(len(uw_blood)) for ii in range(len(D[j])-1)] 
        ww = [8,2,2,2,2,1,1,1,1,1] # new after 15 seconds (loss5 (1-->3)) (loss8 3-->5)
        loss_walls = [ww[i]*tf.reduce_sum([loss_walls_[j] for j in pu.wallGroup[i]])
                                        for i in range(len(pu.wallGroup))]
    return loss_walls, xb_f, uw_skin, uw_blood, duw_blood, D
#---------------------------------------------------------------------------#

def U_Wall_Skin(xwall):
    # BC on walls using the model of 'Skin_3rd' for six blood vessles 
    xwall = tuple(tuple(tf.reshape(xwall[j][i],(-1,pu.d3)) for i in range(len(xwall[j]))) 
                                    for j in range(len(xwall)))  
    Dwall ,Bi, _ = ndds.ndds('Skin_3rd').Bi_blood_wall2(endx = True) 
    N_cum =[]
    for i in range(len(xwall)):
        wall_dim = tf.shape_n(xwall[i])
        dd = [int(jj[0]) for jj in wall_dim] 
        N_cum += [[sum(dd[:i]) for i in range(len(dd)+1)]]
    #-------------------------------------------------------------------------#
    xwall = tuple(tf.concat(xwi ,axis =0) for xwi in xwall)
    
    xb_f = tuple(tf.gather(params = xwall[i], indices = [pu.mDir()[i],3,4]  if c.P_variable 
                else [pu.mDir()[i],3], axis=1) for i in range(len(xwall)))    
    
    T_b, dT_b = (), ()
    i3 = pu.index('Skin_3rd')
    for wii,xw in enumerate(xwall):
        dT_all = ()
        T_all, *gard_T = net.skinMerged((xw,), mixed = 0, T_num = i3)
        for jj in range(len(Dwall[wii])):
            dT_all += (gard_T[Dwall[wii][jj]][i3][N_cum[wii][jj]:N_cum[wii][jj+1],:]/Bi[wii][jj],)
        T_b += (T_all[i3],)
        dT_b +=(tf.concat(dT_all, axis=0),)

    return T_b, dT_b, xb_f, N_cum