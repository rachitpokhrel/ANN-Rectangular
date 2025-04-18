

import Models.ModelG as model_G
import Models.Model as model
import PINN.PINNUtilities as pu
import Domain.Constants as c
import tensorflow as tf

def merge(self, path_):
    #Combine and merge the models
    INPUTS_, OUTPUTS_= () ,()
    for key in self.Tissues_name: #self.Model_keys:
        print('key is: ' ,key)                       
        input_ = tf.keras.Input(shape = (pu.layers['Skin'][0],),dtype = c.DTYPE)
        # Initialize models for triple layers if we don't have saved_model  
        #Act = 'tanh' #if key=='Skin_3rd' else 'swish'
        model_T = model_G.PINN_model(pu.activationFunction, pu.layers['Skin'], LBUB=(pu.LB[key],pu.UB[key]))
        output_ = model_T(input_)
        INPUTS_ += (input_ ,)
        OUTPUTS_ += (output_ ,)
    merged_model = tf.keras.Model(inputs = INPUTS_ , outputs = OUTPUTS_)
        
    if path_:#['3rd']:  
        #W_load1 = tf.keras.models.load_model(path_['1st']+'/tissue', compile=False).get_weights()[:24] 
        #W_load2 = tf.keras.models.load_model(path_['1st']+'/tissue', compile=False).get_weights()[12:24] 
        W_load = tf.keras.models.load_model(path_+'/tissue', compile=False).get_weights()#[24:]
        #W_load = W_load1 + W_load3#+ W_load3 
        print('*************** Tissues Model is loaded*******************')
        merged_model.set_weights(W_load)
    #-----------------------------------------------------------------#
    return merged_model
    

def mergeBlood(self, bloodName, path_blood):
    INPUTS_, OUTPUTS_ = () ,()
    for ii in bloodName:
        print('key is: ' ,ii)
        # Initialize models for blood vessels   
        #layers['Blood'] = [3, 45, 45, 45, 45, 1]
        input_ = tf.keras.Input(shape = (pu.layers['Blood'][0],), dtype = c.DTYPE)                
        Bands_ =(pu.BloodBand['lb'][ii:ii+1 ,[pu.mDir[ii],3]], 
                    pu.BloodBand['ub'][ii:ii+1 ,[pu.mDir[ii],3]])
        model_blood = model.PINN_model(pu.activationFunction, pu.layers['Blood'], LBUB = Bands_)
            
        INPUTS_ += (input_ ,)
        output_ = model_blood(input_)
        OUTPUTS_ += (output_ ,)
    merged_model_blood = tf.keras.Model(inputs = INPUTS_ , outputs = OUTPUTS_ )
    
    try:
        print('***************Blood Model is loaded*******************')
        merged_model_blood.set_weights(tf.keras.models.load_model(path_blood+'/blood', \
                                                    compile=False).get_weights())

    except: tf.print('********The blood models could not be loaded*******')

    return merged_model_blood