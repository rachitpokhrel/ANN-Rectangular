

import PINN.PINNUtilities as pu
import tensorflow as tf
import pickle as pkl



def blood(load_data, rank_ = 0, data_path=[]): 
    if load_data:
        with open(data_path+'/newBloodData_MCMH'+str(rank_)+'_Rank.pkl' , 'rb') as f:
            new_data = pkl.load(f)
        train_dataset_blood = tf.data.Dataset.from_tensors(new_data['X_blood'])  #from_tensors
    else:
        Xend, Xwall = pu.argsBlood()
        #Dw_shape = int(Xwall[0][0].shape[0])
        xwall_out = tuple(pu.fd(xwall_i) for xwall_i in Xwall)
        train_dataset_blood = tf.data.Dataset.from_tensors((pu.fd(Xend), xwall_out))
        #train_dataset_blood =  train_dataset_.batch(Dw_shape, drop_remainder=True)
    train_dataset_blood.prefetch(tf.data.AUTOTUNE)
    return train_dataset_blood

def tissues(load_data, ntime, args_tissue = [], rank_ = 0,data_path=[]):       # Tissues
    #  args_tissue = dict(['X0', 'Xf', 'X_lb', 'X_ub', 'Y_lb_ub', 'Z_lb', 'Z_ub'])
    # Tissue name: ['Skin_1st','Skin_2nd','Tumor1','Tumor2',
    #                   'Gold_Shell', 'Skin_3rd','Skin_3rd_blood']
    if load_data:
        with open(data_path+'/newData_MCMH'+str(rank_)+'_Rank.pkl' , 'rb') as f:
            new_data = pkl.load(f)
        T0 ,T0t = pu.create_T0(new_data['X_tissue'][0], ntime)
        train_dataset = tf.data.Dataset.from_tensors(new_data['X_tissue'][:-2]+(T0,T0t))
            
    else:
        # X0, ,Xf, X_ub, X_lb, Y_lb, Z_lb [all tissues (7)], 
        # Z_ub [skin_3rd, tumor2, gold, skin_3rd_blood]
        X0_p = pu.fd(args_tissue['X0'])
        T0, T0t = pu.create_T0(X0_p, ntime)
        
        dataset = (X0_p, pu.fd(args_tissue['Xf']),  pu.fd(args_tissue['X_lb']),
                        pu.fd(args_tissue['X_ub']), pu.fd(args_tissue['Y_lb_ub']),
                        pu.fd(args_tissue['Z_lb']), pu.fd(args_tissue['Z_ub']),T0 ,T0t)
    
        if 'Xfmoreb' in args_tissue.keys(): dataset+=(pu.fd(args_tissue['Xfmoreb']),)
        train_dataset_ = tf.data.Dataset.from_tensor_slices(dataset)
        train_dataset = train_dataset_.shuffle(args_tissue['X0'][0].shape[0]).\
                                batch(pu.D1_shape,drop_remainder=True)
    train_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset