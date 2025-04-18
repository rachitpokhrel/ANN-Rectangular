import pickle as pkl




# Initialize the class
# 1 - input_args = {'args_tissue':args_tissue,'args_blood': args_blood}
# args_tissue = dict(['X0', 'Xf', 'X_lb', 'X_ub', 'Y_lb_ub', 'Z_lb', 'Z_ub'])
# args_blood = (Xend, xwall)
# 2 - Tissue name: ['Skin_1st','Skin_2nd','Tumor1','Tumor2',
#   'Gold_Shell', 'Skin_3rd','Skin_3rd_blood']
# 3 - layers : {'Skin': [5, 80, 80, 80, 80, 80, 1], 'Blood': [3, 20, 20, 20, 20, 1]}
# 4,5 - lb, ub: dict containing non-dim lb and ub of size 5 [x, y, z, t, P]
# 10,11 - path_ , path_T0 :The pathes the models were saved and should be loaded now 



save_path0 = '/work/???'
save_path0_new = '/work/???'


data_path1 = '/work/???'
with open(data_path1 + 'B_T_newG.pkl', 'rb') as f:
    DataInfo = pkl.load(f)
T0_new = DataInfo['T0_new']
keys = DataInfo['Data_order']
Nf , N_points = DataInfo['Nf'] , DataInfo['All_points'] 
C_band3 = DataInfo['C_band3']
layers = DataInfo['layers']
minpoints = DataInfo['Min_num_ptns'][0]
LB, UB = DataInfo['LB'], DataInfo['UB']
Blood_Band = DataInfo['Blood_Band']
Blood_Band_ND = DataInfo['Blood_Band_ND']
Time = Blood_Band_ND['time'] 

input_args =  {'args_tissue':DataInfo['Tissue_data'],
'args_blood': (DataInfo['X_end'], DataInfo['X_wall'])}


def isCaseNew():
    return True

def isTumorPresent():
    return True

def isBloodPresent():
    return True

def segments():
    return [isTumorPresent(), isBloodPresent()]

def loadWT():#######
    return [False,  True, False, False, False, False, False, False, False]

def loadData():#########
    return [False,  True,  True,  False,   True,  True,  True, True, True]

def loadW(index):#######
    return loadWT()[index], loadData()[index]

def place():#######
    return [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3] 

def cont():#########
    return [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def previousTime():######
    return [[0,1], [0,1], [490,500], [500,510], [510,525], [525,540], [540,555],[555,570],[570,585]]

def nextTime():#########
    return [[0,1], [1,5], [500,510], [510,525], [525,540], [540,555], [555,570],[570,585],[585,600]]

def iterationStop(): ###########
    return [70000 for _  in range(len(nextTime()))]

def Tn():########
    return [1, 2, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]

def PDEw():
    return [50, 50, 20, 25,10, 35] # [1st, 2nd, 3rd, t1, t2, g]

def totalPoints():
    TotalPoints = 0
    for ki in N_points: 
        TotalPoints+=sum(N_points[ki])
        print(f'Total number of points  in {ki} is {sum(N_points[ki])}')
        print('Total number of points are: '  + str(TotalPoints))

iterationStop()[3] = 85000