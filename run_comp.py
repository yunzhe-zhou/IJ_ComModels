from utils import *
from nn_utils import *

# calculate directional derivatives of random forests for model comparisons
N = 1000
d = 6
noise = 1
n_test = 5
s_frac = 0.2
max_depth_ls = [3,5,7,9,"full"]
lh_bias = True
deriv_bias = False

for model_type in ["linear","friedman","constant"]:
    for n_estimators in [1000,5000]:
        result_all = []
        for seed in tqdm(range(200)):
            result_ls = []
            X, Y, X_test, true = sim_gen(seed,N,d,noise,n_test,model_type)
            for max_depth in max_depth_ls:
                result = cal_dir_deriv_RF(X,Y,X_test,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
                result_ls.append(result)
            result_all.append(result_ls)
            open_file = open("data/RF_"+model_type+"_B_"+str(n_estimators)+"_test5.pkl", "wb")
            pickle.dump(result_all, open_file)
            open_file.close()
            
# calculate directional derivatives of random forests for calculate confidence interval and model modifications     
N = 1000
d = 6
noise = 1
n_test = 100
s_frac = 0.2
max_depth_ls = [3,5,7,9,"full"]
lh_bias = True
deriv_bias = True

for model_type in ["linear","friedman","constant"]:
    for n_estimators in [1000,5000]:
        result_all = []
        for seed in tqdm(range(200)):
            result_ls = []
            X, Y, X_test, true = sim_gen(seed,N,d,noise,n_test,model_type)
            for max_depth in max_depth_ls:
                result = cal_dir_deriv_RF(X,Y,X_test,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
                result_ls.append(result)
            result_all.append(result_ls)
            open_file = open("data/RF_"+model_type+"_B_"+str(n_estimators)+"_test100.pkl", "wb")
            pickle.dump(result_all, open_file)
            open_file.close()
            
# calculate directional derivatives of random forests for boosting RF with RF    
N = 1000
d = 6
noise = 1
n_test = 100
s_frac = 0.2
max_depth_ls = [3,5,7,9,"full"]
lh_bias = False
deriv_bias = False

for model_type in ["linear","friedman","constant"]:
    for n_estimators in [1000,5000]:
        result_all = []
        for seed in tqdm(range(200)):
            result_ls = []
            X, Y, X_test, true = sim_gen(seed,N,d,noise,n_test,model_type)
            k= 0 
            result0 = cal_dir_deriv_RF(X,Y,X_test,n_estimators,int(s_frac*N),max_depth_ls[k],lh_bias,deriv_bias)
            pred_step1 =  result0[5]
            for max_depth in max_depth_ls:
                result = cal_dir_deriv_RF(X,Y - pred_step1,X_test,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
                result_ls.append([result0,result])
            result_all.append(result_ls)
            open_file = open("data/RF_boost_"+model_type+"_B_"+str(n_estimators)+"_test100.pkl", "wb")
            pickle.dump(result_all, open_file)
            open_file.close()       
            
# calculate directional derivatives of random forests for boosting RF with glm             
N = 1000
d = 6
noise = 1
n_test = 100
s_frac = 0.2
max_depth_ls = [3,5,7,9,"full"]
lh_bias = False
deriv_bias = False

for model_type in ["linear","friedman","constant"]:
    for n_estimators in [1000,5000]:
        result_all = []
        for seed in tqdm(range(200)):
            result_ls = []
            X, Y, X_test, true = sim_gen(seed+len(result_all),N,d,noise,n_test,model_type)
            Sigma, U, pred_step1,pred_test = LM_pred(X,X_test,Y,N)
            result0 = [Sigma, U, pred_step1,pred_test]
            for max_depth in max_depth_ls:
                result = cal_dir_deriv_RF(X,Y - pred_step1,X_test,n_estimators,int(s_frac*N),max_depth,lh_bias,deriv_bias)
                result_ls.append([result0,result])
            result_all.append(result_ls)
            open_file = open("data/LM_boost_"+model_type+"_B_"+str(n_estimators)+"_test100.pkl", "wb")
            pickle.dump(result_all, open_file)
            open_file.close()            
            
# calculate directional derivatives of glm for model comparisons            
N = 1000
d = 6
noise = 1
n_test = 5

for model_type in ["linear","friedman","constant"]:
    pred_ls = []
    Sigma_ls = []
    U_ls = []
    for seed in range(200):
        X, Y, X_test, true = sim_gen(seed,N,d,noise,n_test,model_type)
        X = np.concatenate([X,np.ones(N).reshape([-1,1])],axis=1)
        X_test = np.concatenate([X_test,np.ones(n_test).reshape([-1,1])],axis=1)
        XtX = np.matmul(X.T,X)
        theta = np.matmul(np.linalg.inv(XtX),np.matmul(X.T,Y))
        D1 = X_test
        H = 2*XtX/N
        D2 = 2 * X * (Y - np.matmul(X,theta))
        U = np.matmul(np.matmul(D1,np.linalg.inv(H)),D2.T)
        Sigma = np.matmul(U,U.T)/(N**2)
        pred = np.matmul(X_test,theta)
        pred_ls.append(pred)
        Sigma_ls.append(Sigma)
        U_ls.append(U)
    result_all = [[Sigma_ls[i],U_ls[i],pred_ls[i]] for i in range(len(U_ls))]

    open_file = open("data/LM_"+model_type+"_test5.pkl", "wb")
    pickle.dump(result_all, open_file)
    open_file.close()

# calculate directional derivatives of neural network for comparisons 
N = 1000
d = 6
n_test = 5
h_dim = 5
EPOCH = 1000
lr = 0.01
act = 'sigmoid'
epoch_choice = [1000]
lam_choice = [1e-03]
cond_shred_choice = [100000]
n_iter = 200

for model_type in ["linear","friedman","constant"]:
    result_all = []
    for seed in tqdm(range(n_iter)):
        result = simulate_NN(seed,N,d,n_test,h_dim,EPOCH,lr,act,model_type,epoch_choice,lam_choice,cond_shred_choice)
        result_all.append(result)
        open_file = open("data/NN_"+model_type+"_test5.pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()

# calculate directional derivatives of XGBoost for comparisons
N = 1000
d = 6
noise = 1
n_test = 5
s_frac = 0.2
n_estimators = 1000

for model_type in ["linear","friedman","constant"]:
    result_all = []
    for seed in tqdm(range(200)):
        result_ls = []
        X, Y, X_test, true = sim_gen(seed,N,d,noise,n_test,model_type)
        result = cal_dir_deriv_XGB(X,Y,X_test,n_estimators,int(s_frac*N))
        result_all.append(result)
        open_file = open("data/XGB_"+model_type+"_B_"+str(n_estimators)+"_test5.pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()

# calculate directional derivatives of XGBoost for comparisons
N = 1000
d = 6
noise = 1
n_test = 100
s_frac = 0.2
n_estimators = 1000

for model_type in ["linear","friedman","constant"]:
    result_all = []
    for seed in tqdm(range(200)):
        result_ls = []
        X, Y, X_test, true = sim_gen(seed,N,d,noise,n_test,model_type)
        result = cal_dir_deriv_XGB(X,Y,X_test,n_estimators,int(s_frac*N))
        result_all.append(result)
        open_file = open("data/XGB_"+model_type+"_B_"+str(n_estimators)+"_test100.pkl", "wb")
        pickle.dump(result_all, open_file)
        open_file.close()