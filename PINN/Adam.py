

import PINN.PINN as pinn
import PINN.AdamUtilities as au
import time
        

def adam(self, index):
    au.Blood_Band_ND["time"] = au.nextTime()()[index]
    if index == 1: 
        path_T0 = au.save_path0 + str(au.previousTime()()[index][0])+'s_'+str(au.previousTime()()[index][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand'
    else: path_T0 = au.save_path0_new + str(au.previousTime()()[index][0])+'s_'+str(au.previousTime()()[index][1]) + 's_B_T_newG2_b3'

    LL_Loss = 4.0e2
    mcmh = True
    alpha_constant=False

    path_ = path_T0
    path_blood = path_


    if au.cont()[index]:
        path_ = au.save_path0 + str(au.nextTime()()[index][0])+'s_'+str(au.nextTime()()[index][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand_max2_2'
        LL_Loss = 5.0e2
        mcmh = False
        alpha_constant=True
        path_blood = path_


    data_path = au.save_path0 + str(au.previousTime()()[index][0])+'s_'+str(au.previousTime()()[index][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand_max2_2'
    save_path = au.save_path0 + str(au.nextTime()()[index][0])+'s_'+str(au.nextTime()()[index][1]) + 's_B_T_newG2_tanh_softmax_mcmh_rand_max2_2'#_adamw'

    start_time = time.time() 
    load_w_data =(au.loadWT()[index],au.loadData()[index])
    
    ntime = 4 if index == 3 else 6

    ee= 2.0e-4

    pinn.load_w_data = load_w_data
    pinn.path_ = path_
    pinn.path_T0 = path_T0
    pinn.ntime = ntime
    pinn.it_Stop = au.iterationStop[index]
    pinn.lr_cycle_start = au.place[index]
    pinn.path_blood = path_blood
    pinn.tn = au.Tn[index]
    
    lossT = pinn.train(save_path = save_path, 
                        max_Stop_err = ee, 
                        mcmh = mcmh,
                        data_path = data_path,  
                        LL_Loss = LL_Loss, 
                        opt = 'parallel',
                        it_Stop_lbfgs=0,
                        alpha_constant=alpha_constant)

    

