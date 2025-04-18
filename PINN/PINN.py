import PINN.AdamUtilities as au
import PINN.PINNUtilities as pu
import PINN.Datasets as d
import PINN.MergedModel as mm
import PINN.LRSchedule as LRSchedule
import PINN.Alpha as alpha
import PINN.MHA as mha
import PINN.LossSkin as ls
import PINN.LossBlood as lb
import tensorflow as tf
import numpy as np
import pickle as pkl
import os
import time

# Initialize Horovod --> Step2 " 3- Configure CPU-GPU Mapping
import horovod.tensorflow as hvd    # Step 1
hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: # Memory growth must be set before GPUs have been initialized
   tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
print(len(gpus), "Physical GPUs with hvd size: ",hvd.size(), " & hvd rank: ", hvd.rank())

load_w_data = (False, False)
path_ = ""
path_T0 = ""
ntime = 0
iterationStop = au.iterationStop[0]
LRCycleStart = au.place[0]
path_blood = ""
tn = 0

def train(self, save_path = [], max_Stop_err = 2.00e-5, mcmh = False, data_path = [], LL_Loss =650, opt = 'parallel',it_Stop_lbfgs = 0,alpha_constant = 'False'): 
    #-------------------------------------------------------------------------#    
    # initializations
    if not save_path: save_path = path_
    if not data_path: data_path = path_T0
    it_Stop = it_Stop

    it , stopping_run   = 0, 0
    LL_Loss = LL_Loss
    losses_lbfgs = []

    region = 'Tumor_blood' if (pu.isBloodPresent and pu.isTumorPresent) else 'Tumor' if pu.isTumorPresent else 'Skin-blood'
    #--------------------------- indices if Losses ---------------------------#
    Nt = len(pu.keys)
    # f |0 | ubx | lbx |y
    Loss_ind = {'Skin_1st':[0, Nt,   2*Nt,   3*Nt,   4*Nt,   4*Nt+3],
                'Skin_2nd':[1, Nt+1, 2*Nt+1, 3*Nt+1, 4*Nt+1 ,4*Nt+4],
                'Skin_3rd':[2, Nt+2, 2*Nt+2, 3*Nt+2, 4*Nt+2, 4*Nt+5]}
    ind_n , Ifc_n = 4*Nt+6 ,6

    if pu.isTumorPresent: 
        Loss_ind['Tumor1'] = [3, Nt+3, 2*Nt+3, 3*Nt+3, ind_n,ind_n+3,ind_n+6,ind_n+9]
        Loss_ind['Tumor2'] = [i+1 for i in Loss_ind['Tumor1']]
        Loss_ind['Gold_Shell'] = [i+1 for i in Loss_ind['Tumor2']]
        ind_n +=12
        Ifc_n +=10
    
    Loss_ind['IFC'] = list(range(ind_n, Ifc_n + ind_n))

    if pu.isBloodPresent: 
        ind_b = Ifc_n+ind_n
        b_in = 13
        Loss_ind['Blood_PDE'] = list(range(ind_b, 6 + ind_b))
        Loss_ind['Blood_in'] = list(range(6 + ind_b, 12 + ind_b))
        Loss_ind['Blood_out_sys_endx'] = list(range(12 + ind_b, b_in + ind_b))
        Loss_ind['Blood_wall'] = list(range(b_in + ind_b,b_in + ind_b + len(pu.wallGroup())))   
    #-------------------------------------------------------------------------#
    if hvd.rank()==0:
        writer_train = tf.summary.create_file_writer(save_path + 'train')
        os.makedirs(save_path,exist_ok=True)
        start_time = time.time()
    #-------------------------------------------------------------------------#
    # 4- Distribute Data and initialize NN
    for i in range(hvd.size()):
        if hvd.rank()==i:
            if load_w_data[0]:#load_w_data[1]: 
                with open(data_path+'/Power'+str(hvd.rank())+'_Rank.pkl','rb') as f:
                    P_W_Data = pkl.load(f)
                
                Losses = P_W_Data['Loss_far']
                pu.w_t = P_W_Data['wL']['wt']
                if pu.isBloodPresent:
                    pu.w_wall, pu.w_b = P_W_Data['wL']['wwall'], P_W_Data['wL']['wb']
        #-----------------------------------------------------------------#
            if load_w_data[1]:
                train_dataset = d.tissues(rank_ = i, data_path=data_path)
                XT_tf = next(iter(train_dataset)) 
            else:
                XT_tf = next(iter(train_dataset.skip(i).take(1)))
        #-----------------------------------------------------------------#    
            print(f'************I am in rank {i}********************')
            if pu.isBloodPresent:
                if load_w_data[1]:
                    train_dataset_blood = d.blood(rank_ = i,data_path=data_path)
                XB_tf = next(iter(train_dataset_blood))
            else: XB_tf = [] 
    #-----------------------------------------------------------------#
    step_LR = int(2*(pu.learningRateCycleStart+4))
    lossTotal = 2 
    weight_ = pu.w_t + pu.w_b + pu.w_wall  if pu.isBloodPresent else pu.w_t
    it_stop = min(60000,int(it_Stop*0.45))
    Loss_num = len(weight_) 
    #---------------------------------------------------------------------#
    while stopping_run<=10002 and it<it_Stop :

        if it==0: 
            Loss, W_Loss = loss_fn_opt(XT_tf, XB_tf, first_batch = True, opt = opt) 
        else: 
            Loss, W_Loss = loss_fn_opt(XT_tf, XB_tf, first_batch = False, opt = opt) 
    #----------------------------------------------------------------------#
    if it<it_stop and lossTotal<0.2 and hvd.rank()==0:
        if mcmh and it%200==0: 
            XT_tf = mha.main(XT_tf)
            if pu.isBloodPresent:
                XB_tf = mha.wall(XB_tf)
    #---------------------------------------------------------------------#
    Losses = np.reshape(Loss,(Loss_num,1))
    W_Loss_t = np.reshape(W_Loss,(Loss_num,))
    lossTotal = hvd.allreduce(np.sum(Losses), average=True)              
    #%%--------------------------------------------------------------------#
    # weight optimizations 
    if (it>it_Stop*0.1 and lossTotal<0.1) or load_w_data[0]: 
        if it%1==0:# and it<it_Stop*0.8: 
            dwPINN_update(tf.convert_to_tensor(Losses), weight_) 
    #%%--------------------------------------------------------------------#
    pu.step0.assign_add(1)
    if it>it_Stop*0.92 or stopping_run>0: max_lr, base_lr = 1, 1
    else:
        if pu.LRMethod =='cycle':
            if it%10000==0 and (load_w_data[0] or it>0):
                if it<=it_Stop*0.06:# or it>=it_Stop*0.72: 
                    alpha.alpha_max_min(load_w_data[0],W_Loss_t, it, LL_Loss)
            if it%1000==0 and (load_w_data[0] or it>0):
                if it>it_Stop*0.06: 
                    alpha.alpha(load_w_data[0],W_Loss_t, it, stopping_run, step_LR, LL_Loss)
                
            if it<it_Stop*0.06: max_lr = 0
            elif it>it_Stop*0.82: max_lr = 0.5  # 8e-6
            elif it>it_Stop*0.74: max_lr = 0.25 # 1e-5
        else:
            if it>=2200 and it%1000==0: alpha.alpha(load_w_data[0], W_Loss_t, it, stopping_run,step_LR, LL_Loss )

    if alpha_constant: 
        if it>it_Stop*0.6: max_lr, base_lr = 1, 1
        else: max_lr = 0.5
    learning_rate = LRSchedule.LRSchedule(pu.step0 , max_lr, base_lr)
    pu.optimizer_method.learning_rate = learning_rate 
    if lossTotal<max_Stop_err and it>it_Stop*0.25: stopping_run+=1
    #%%--------------------------------------------------------------------#
    if hvd.rank() == 0:
        if  it % 500 == 0:
            elapsed = time.time() - start_time
            tf.print('rank0: %s-Region,- %d s- ,It: %d, Loss: %.3e, Loss overal:%.3e, w-Loss: %.3e, Time: %.2f, lr: %.3e' 
                    %(region, pu.Time[1], it, np.sum(Losses),lossTotal,np.sum(W_Loss_t),
                            elapsed, learning_rate.numpy()))
            if  it % 5000 == 0:
                for iw in [2,5]:
                    tf.print( f' weight {iw} is {weight_[iw]}****')
            start_time = time.time()
        #--------------------------Print Losses----------------------------#
            for index in Loss_ind.keys():
                tf.print('loss_ %s: %.3e'% (index,sum([Losses[i,0] for i in Loss_ind[index]])))                    
            #--------------------Save the summary of Results----------------#      
            with writer_train.as_default():  
                for i in range(len(Losses)):
                    tf.summary.scalar(name='Loss_' + str(i), data=Losses[i,0], step=it)
                    tf.summary.scalar(name='Weight_'+ str(i), data=weight_[i], step=it)
        if  it % 10 == 0: 
            with writer_train.as_default():
                tf.summary.scalar(name ='iter', data = it, step=it)
                tf.summary.scalar(name ='loss_Total', data = lossTotal, step=it)
                tf.summary.scalar(name ='learning_rate', data = learning_rate, step=it)
        #---------------------------------------------------------------#
        # Save Models
        if it %2500 == 0 or it==it_Stop-1 or stopping_run==10002:
            mm.merge.save(save_path+'/tissue',save_traces=True)
                                #save_format='keras_v3')#, 
            if pu.isBloodPresent!=0:
                mm.mergeBlood.save(save_path+'/blood', save_traces=True)#save_format='keras_v3'
    #------------------------ Save for  all ranks--------------------------
    if (it%2000==0 and it>3000) or  it==it_Stop-1 or stopping_run==10002:
        if it<it_stop:
            with open(save_path+'/newData_MCMH'+str(hvd.rank())+'_Rank.pkl','wb') as f:
                pkl.dump({'X_tissue': XT_tf },f)
            if pu.isBloodPresent: 
                with open(save_path+'/newBloodData_MCMH'+str(hvd.rank())+'_Rank.pkl','wb') as f:
                    pkl.dump({'X_blood':XB_tf},f)
        #------------------------------------------------------------------#
        with open(save_path+'/Power'+str(hvd.rank())+'_Rank.pkl','wb') as f:
            if pu.isBloodPresent:
                Weight_save = {'wt':pu.w_t , #weight_[:len(pu.w_t)],
                                'wwall':pu.w_wall,# weight_[-len(pu.w_wall):],
                                'wb':pu.w_b}#weight_[len(pu.w_t):len(pu.w_t)+len(pu.w_b)]}
            else: Weight_save = {'wt': pu.w_t}
            
            pkl.dump({'wL': Weight_save,'Loss_far':Losses,'LL_Loss':LL_Loss},f)
    #----------------------------------------------------------------------#        
    hvd.allreduce(tf.constant(0), name="Barrier")  #-------
    if it%2000==0: tf.print(f' hvd rank = { hvd.rank()} and iter is {it}')
    it+=1
    return lossTotal

@tf.function
def dwPINN_update(self, Losses , weight_ ):
    with tf.GradientTape(persistent=True) as tapew:
        tapew.watch(weight_)
        WL= [-Losses[i]*weight_[i] for i in range(len(weight_))]
    
    grads_wt = tapew.gradient(WL , weight_)
    pu.optimizer_Wt.apply_gradients(zip(grads_wt, weight_))


@tf.function#(jit_compile = True)  
def loss_fn_opt(self, X_dist_Tissue, X_blood = [], first_batch = False , opt ='sequential'):  
    with tf.GradientTape(persistent=True) as tape:
        Losses_T = ls.Tissues(X_dist_Tissue)
        W_Losses_T = [pu.w_t[i]*Losses_T[i] for i in range(len(pu.w_t))]
        #--------------------------------------------------------------------#
        if pu.isBloodPresent:
            Losses_B, Losses_W = lb.Vessels(X_blood)
            W_Losses_B = [pu.w_b[i]*Losses_B[i] for i in range(len(pu.w_b))]
            W_Losses_W = [pu.w_wall[i]*Losses_W[i] for i in range(len(pu.w_wall))]
        else:
            W_Losses_B ,Losses_B, Losses_W, W_Losses_W = [], [], [], []

        W_Losses = W_Losses_T + W_Losses_B + W_Losses_W
        Losses = Losses_T + Losses_B + Losses_W
        wLoss = tf.reduce_sum(W_Losses)

    