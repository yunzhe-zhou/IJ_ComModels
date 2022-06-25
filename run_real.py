import numpy as np
from utils import *
from nn_utils import *
from tqdm import tqdm

# calculate directional derivatives of random forests for CI
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv')
N = X_train_split[0].shape[0]
print(N)
s_frac = 0.2
max_depth_ls = ["full"]
lh_bias = True
deriv_bias = True
for n_estimators in [1000]:
    result_all = []
    for seed in tqdm(range(50)):
        np.random.seed(seed)
        result_ls = []
        X = X_train_split[seed]
        Y = y_train_split[seed].reshape([-1,1])
        for max_depth in max_depth_ls:
            result,pred_test_all = cal_dir_deriv_RF_real(X,Y,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
            result_ls.append([result,pred_test_all])
        result_all.append(result_ls)
        open_file = open("data/real_data_RF_"+"B_"+str(n_estimators)+".pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()

# calculate directional derivatives of random forests for boosting LM with RF    
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv')
N = X_train_split[0].shape[0]
print(N)
s_frac = 0.2
max_depth_ls = ["full"]
lh_bias = False
deriv_bias = False
for n_estimators in [1000]:
    result_all = []
    for seed in tqdm(range(50)):
        np.random.seed(seed)
        result_ls = []
        X = X_train_split[seed]
        Y = y_train_split[seed].reshape([-1,1])
        Sigma, U, pred_step1,pred_test,pred_test_all0 = LM_pred_real(X,X_test,X_test_all,Y,N)
        result0 = [Sigma, U, pred_step1,pred_test]
        for max_depth in max_depth_ls:
            result,pred_test_all = cal_dir_deriv_RF_real(X,Y - pred_step1,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
            result_ls.append([result0,result,pred_test_all0,pred_test_all])
        result_all.append(result_ls)
        open_file = open("data/real_data_LM_boost_"+"B_"+str(n_estimators)+".pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()  

# calculate directional derivatives of random forests for boosting RF with RF    
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv')
N = X_train_split[0].shape[0]
print(N)
s_frac = 0.2
max_depth_ls = ["full"]
lh_bias = False
deriv_bias = False
for n_estimators in [1000]:
    result_all = []
    for seed in tqdm(range(50)):
        np.random.seed(seed)
        result_ls = []
        X = X_train_split[seed]
        Y = y_train_split[seed].reshape([-1,1])
        k= 0 
        result0,pred_test_all0 = cal_dir_deriv_RF_real(X,Y,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth_ls[k],lh_bias,deriv_bias)
        pred_step1 =  result0[5]
        for max_depth in max_depth_ls:
            result,pred_test_all = cal_dir_deriv_RF_real(X,Y - pred_step1,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
            result_ls.append([result0,result,pred_test_all0,pred_test_all])
        result_all.append(result_ls)
        open_file = open("data/real_data_RF_boost_"+"B_"+str(n_estimators)+".pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()  

# calculate directional derivatives of XGBoost for comparisons
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv') 
N = X_train_split[0].shape[0]
s_frac = 0.2
n_estimators = 1000

result_all = []
for seed in tqdm(range(50)):
    np.random.seed(seed+100)
    result_ls = []
    X = X_train_split[seed]
    Y = y_train_split[seed].reshape([-1,1])
    result,pred_test_all = cal_dir_deriv_XGB_real(X,Y,X_test,X_test_all,n_estimators,int(s_frac*N))
    result_all.append([result,pred_test_all])
    open_file = open("data/real_data_XGB_"+"B_"+str(n_estimators)+".pkl", "wb")
    pickle.dump(result_all, open_file)
    open_file.close()


    # calculate directional derivatives of glm for model comparisons    
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv')        
N = X_train_split[0].shape[0]
X_test = np.concatenate([X_test,np.ones(X_test.shape[0]).reshape([-1,1])],axis=1)
X_test_all = np.concatenate([X_test_all,np.ones(X_test_all.shape[0]).reshape([-1,1])],axis=1)

pred_ls = []
Sigma_ls = []
U_ls = []
pred_test_all_ls = []
for seed in tqdm(range(50)):
    result_ls = []
    X = X_train_split[seed]
    Y = y_train_split[seed].reshape([-1,1])
    X = np.concatenate([X,np.ones(N).reshape([-1,1])],axis=1)
    XtX = np.matmul(X.T,X)
    theta = np.matmul(np.linalg.pinv(XtX),np.matmul(X.T,Y))
    D1 = X_test
    H = 2*XtX/N
    D2 = 2 * X * (Y - np.matmul(X,theta))
    U = np.matmul(np.matmul(D1,np.linalg.pinv(H)),D2.T)
    Sigma = np.matmul(U,U.T)/(N**2)
    pred = np.matmul(X_test,theta)
    pred_test_all = np.matmul(X_test_all,theta)
    pred_ls.append(pred)
    Sigma_ls.append(Sigma)
    U_ls.append(U)
    pred_test_all_ls.append(pred_test_all)
result_all = [[[Sigma_ls[i],U_ls[i],pred_ls[i]],pred_test_all_ls[i]] for i in range(len(U_ls))]

open_file = open("data/real_data_LM"+".pkl", "wb")
pickle.dump(result_all, open_file)
open_file.close()


# calculate directional derivatives of random forests for model comparisons
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv')
N = X_train_split[0].shape[0]
print(N)
s_frac = 0.2
max_depth_ls = [3,5,7,9,"full"]
lh_bias = False
deriv_bias = False
for n_estimators in [1000]:
    result_all = []
    for seed in tqdm(range(50)):
        np.random.seed(seed)
        result_ls = []
        X = X_train_split[seed]
        Y = y_train_split[seed].reshape([-1,1])
        for max_depth in max_depth_ls:
            result,pred_test_all = cal_dir_deriv_RF_real(X,Y,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
            result_ls.append([result,pred_test_all])
        result_all.append(result_ls)
        open_file = open("data/real_data_RF_comp_"+"B_"+str(n_estimators)+".pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()

# calculate directional derivatives of random forests for boosting LM with RF    
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv')
N = X_train_split[0].shape[0]
print(N)
s_frac = 0.2
max_depth_ls = ["full"]
lh_bias = False
deriv_bias = False
for n_estimators in [1000]:
    result_all = []
    for seed in tqdm(range(50)):
        np.random.seed(seed+200)
        result_ls = []
        X = X_train_split[seed]
        Y = y_train_split[seed].reshape([-1,1])
        Sigma, U, pred_step1,pred_test,pred_test_all0 = LM_pred_real(X,X_test,X_test_all,Y,N)
        result0 = [Sigma, U, pred_step1,pred_test]
        for max_depth in max_depth_ls:
            result,pred_test_all = cal_dir_deriv_RF_real(X,Y - pred_step1,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
            result_ls.append([result0,result,pred_test_all0,pred_test_all])
        result_all.append(result_ls)
        open_file = open("data/real_data_LM_boost_"+"B_"+str(n_estimators)+"_new.pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()  


# calculate directional derivatives of random forests for boosting RF with RF    
X_train_split, y_train_split, X_test_all, y_test, X_test = get_real('data/house.csv')
N = X_train_split[0].shape[0]
print(N)
s_frac = 0.2
max_depth_ls = ["full"]
lh_bias = False
deriv_bias = False
for n_estimators in [1000]:
    result_all = []
    for seed in tqdm(range(50)):
        np.random.seed(seed+300)
        result_ls = []
        X = X_train_split[seed]
        Y = y_train_split[seed].reshape([-1,1])
        k= 0 
        result0,pred_test_all0 = cal_dir_deriv_RF_real(X,Y,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth_ls[k],lh_bias,deriv_bias)
        pred_step1 =  result0[5]
        for max_depth in max_depth_ls:
            result,pred_test_all = cal_dir_deriv_RF_real(X,Y - pred_step1,X_test,X_test_all,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
            result_ls.append([result0,result,pred_test_all0,pred_test_all])
        result_all.append(result_ls)
        open_file = open("data/real_data_RF_boost_"+"B_"+str(n_estimators)+"_new.pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()  