
import PINN.PINNUtilities as pu
import PINN.OptimizerMethod as om
import tensorflow as tf
import numpy as np

        
# alpha
def alpha_max_min(self, load_w_t, W_Loss_t, it, LL_Loss):
    # Every 20K iterations --> max and min will change--> e.g.LL_Loss = 200
    # starting 20 K itearion (start is based on our definitions in cycles)
    self.step0.assign(0)
    # m2 = 2000
    ER = np.sum(W_Loss_t)/ LL_Loss
    m1 = 1.1 if ER>1e-4 else 1.04 if ER>2e-5 else 1.01
    if it<=20000 and load_w_t == False: m1 = 1.13
    if it>0: LL_Loss = LL_Loss * m1**5  # each 20K iterations
    R_L = np.sum(W_Loss_t)/ LL_Loss
    self.max_lr, self.base_lr = tf.reduce_min([R_L,6e-4]) , tf.reduce_min([R_L/10,8e-5])
    tf.print(f'****it is {it} |  max_lr is {self.max_lr} and min_lr is {self.base_lr}********')
    
#%%-------------------------------------------------------------------------#    
def alpha(self, load_w_t, W_Loss_t, it, stopping_run, step_LR, LL_Loss):
    #%% Define constants
    max_opt = 22000 if pu.isBloodPresent() else 24000
    ER = np.sum(W_Loss_t)/LL_Loss
    m1 = 1.1 if ER>1e-4 else 1.04 if ER>1e-5 else 1.01 
    m2 = 2000 if ER>8.5e-6 else 4000
    m3 = 1.13

    decay_rates = [1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 
                    1e-4, 9e-5 ,8e-5, 7e-5, 6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 
                    1e-5, 9e-6, 8e-6, 7e-6, 6e-6, 5e-6, 4e-6, 3e-6]
    #%%----------------------------initialization--------------------------#
    self.step_LR_pre = step_LR
    #%%---------------------------------------------------------------------#
    if stopping_run: 
        self.step_LR = 20
    else:   
        if it%2000==0 and it<max_opt and load_w_t==False:
            self.LL_Loss = LL_Loss*m3
        elif it%m2==0:
            self.LL_Loss = LL_Loss*m1
        #-----------------------------------------------------#   
        R_L = np.sum(W_Loss_t)/ self.LL_Loss
        LR_prpos=np.abs(np.array(decay_rates)-R_L)
        if it%10000==0 and it>30000 and om.OptimizerMethod().method(pu.SGD(), pu.optimizer()).learning_rate<1e-4:
            LR_prpos=np.abs(np.array(decay_rates)-2*R_L)
        #-----------------------------------------------------------#
        try: self.step_LR = np.where(LR_prpos==np.min(LR_prpos))[0][0] 
        except: self.step_LR = 20 # 8e-6

        if abs(self.step_LR_pre-self.step_LR)>2:#3100:#2100: # more than two step
            if self.step_LR_pre-self.step_LR<0: self.step_LR = self.step_LR_pre+2 #1500
            else: self.step_LR = self.step_LR_pre-2#1500                  
            if it>20000 and self.step_LR < 3:self.step_LR = 3
        self.max_lr, self.base_lr = decay_rates[self.step_LR],decay_rates[self.step_LR] 
        self.step0.assign(0)
        #--------------------------------------------------------------------#