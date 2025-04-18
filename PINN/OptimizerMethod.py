
import tensorflow as tf

def method(self, SGD, opt):
    if SGD:
        return tf.keras.optimizers.SGD(learning_rate = 1e-4, momentum=0.9)
    if opt=='adamw':
        print('*************** Adam WWWWWWWWWW*******************')

        return tf.keras.optimizers.AdamW(learning_rate=self.learning_rate_schedule(0, self.max_lr, self.base_lr),
                                    amsgrad=True, weight_decay = 1e-4)
    elif opt=='adadelta':
        print('***************Adam Delta*******************')

        return tf.keras.optimizers.Adadelta(learning_rate=0.0002)
    
    else:
        return tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule(0, self.max_lr, self.base_lr))
        
    
def methodWeight(self):                                                     
    # Global weights optimizer  lr_wt = init_lr *0.97**(it/1000)

    #self.optimizer_method.build(self.train_vars())
    #init_lr = 0.001 if self.load_w_t else 0.01
    #self.lr_wt = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = init_lr, 
    #                                                       decay_steps=1000, decay_rate=0.99)

    #self.optimizer_Wt = tf.keras.optimizers.Adam(learning_rate = 0.005)#self.lr_wt) 
    
    print('***************SGD optimizer for weights*******************')
    return tf.keras.optimizers.SGD(learning_rate = 0.99)
