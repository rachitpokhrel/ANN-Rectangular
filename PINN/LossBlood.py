
import Domain.Constants as c
import PINN.PINNUtilities as pu
import tensorflow as tf
import PINN.LossWall as lw
import PINN.BloodProperties as bp
import PINN.Net as net
import Domain.NDDomains as ndds
        

def IC_BCL(Xend): 
    xb_in, xb_out , xend= create_data_in_out(Xend)

    ubin_pred, ubin_l_pred = net.bloodMerged(xb_in)
    ubout_pred, ubout_l_pred = net.bloodMerged(xb_out)
    # Loss 
    # 1 - symmetry loss : in blood levels 1 : input for Arterial and output for venus
    lossSym = tf.reduce_mean(tf.square(ubin_l_pred[1]))+\
                tf.reduce_mean(tf.square(ubout_l_pred[4]))          
    # output of Arterial an level 3 ==Temp output blood at level 3
    u3out_pred = net.skinMerged((xend[0],), mixed = 2, T_num = pu.keys.index('Skin_3rd'))
    LossTout = tf.reduce_mean(tf.square(ubout_pred[2]*pu.dTb_Tt - u3out_pred)) 
    #-----------------------------------------------------------------------# 
    # 2) loss_in: MSE BC in at l = 0 of Blood_vessels
    # Tin for blood vessels based on the Tout of previous vessels
    loss_in = []
    Tinput = tf.constant(ndds.T_in(), dtype=c.DTYPE) 
    for ii in bp.blood_name: 
        if ii== 0: 
            Tin = Tinput
        elif ii==5:# u5in_pred
            Tin = net.skinMerged((xend[1],), mixed = 2, T_num = pu.keys.index('Skin_3rd'))
            Tin = Tin/pu.dTb_Tt
        else:
            Tin = ubout_pred[ii-1] if ii<3 else ubout_pred[ii+1]
        loss_in.append(tf.reduce_mean(tf.square(ubin_pred[ii]-Tin)))

    return loss_in, [LossTout], [lossSym]

def create_data_in_out(xend):
    #LB, UB = blood_band['lb'][0][0,1],blood_band['ub'][0][0,1]
    #time_rand = LB + (UB-LB)*tf.random.uniform([N0,1],dtype =DTYPE)
    N0 =  xend[0].shape[0]
    t_P_rand = xend[0][:,3:]
    
    Xend = tuple(tf.concat([xx[:,:3],t_P_rand], axis =1) for xx in xend)
    Xin = tuple(tf.concat([xin *tf.ones([N0,1],dtype =c.DTYPE),t_P_rand], axis =1) for xin in bp.in_b) 
    Xout = tuple(tf.concat([xout *tf.ones([N0,1],dtype =c.DTYPE),t_P_rand], axis =1) for xout in bp.out_b)
    return Xin,Xout,Xend

def PDE(self, u_wall, ub, ub_dL,plt_loss=0):
    # u_Wall, ub, ub_dL : estimated temperatures in blood vessels and on their walls 
    Loss_f = []
    wb = [5,2,2,1,1,1]  # it was [2,1,2,6,3,1] before 15 seconds
    for j in bp.blood_name:
        factorm = ndds.ndds('Skin_3rd').param.factor_blood(j)
        L0 = ndds.ndds('Skin_3rd').L0_blood_basedOn3rd(j)
        L = tf.constant(L0[1],dtype=c.DTYPE) 
        f_u_avr = ub_dL[j] - factorm[0] *L*(u_wall[j]/pu.dTb_Tt - ub[j])

        if j ==2: #if (m =m_level-1 and key == 'Arterial')
            f_u_avr = f_u_avr -factorm[1]*L*(ub[j] - ndds.ndds('Skin_3rd').ND_T(0,blood=True))
        # 0 - Loss of PDE
        if plt_loss:
            Loss_f.append(tf.square(f_u_avr))
        else:
            Loss_f.append(wb[j]*tf.reduce_mean(tf.square(f_u_avr)))

    return Loss_f  # Loss_adpt_b[0-6]

def Vessels(X_dist_Blood=[]): # blood vessels are located in the "Skin-3rd"
    # xend: (Arterial-out-2(0), Venus-in-2(1))
    xend, xwall = X_dist_Blood
    #-------------------------------------------------------------------------#
    # Wall Data --> xbs, uwall at skin(xbf) and u wall at blood(xbf)
    loss_walls, xb_f, uwall_skin, uw_blood, duw_blood,_ = lw(xwall)

    # Estimating the loss values
    loss_f = PDE(uwall_skin, uw_blood, duw_blood)    #losses [0-5]
    # Create other Datasets here    
    loss_in, LossTout, lossSym = IC_BCL(pu.ft(xend,pu.d3))     
    Losses = loss_f + loss_in + LossTout  #+ lossSym # + loss_endx

    return Losses, loss_walls