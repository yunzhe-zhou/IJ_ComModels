import numpy as np
import random
from tqdm import tqdm
import pickle
import copy
from scipy import stats
from numpy import linalg as LA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def sim_gen(seed,N,d,noise,n_test,model_type):
    np.random.seed(seed)
    X = np.random.uniform(-1,1,N*d).reshape([N,d])

    if model_type == "friedman":
        Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10 * X[:,3] + 5 * X[:,4] + np.random.normal(0,noise,N)).reshape([-1,1])
    elif model_type == "linear":
        Y = (np.sum(X[:,:4],1)+ np.random.normal(0,noise,N)).reshape([-1,1])
    elif model_type == "constant":
        Y = (np.ones(N)*2 + np.random.normal(0,noise,N)).reshape([-1,1])
        
    np.random.seed(2021)
    X_test = np.random.uniform(-1,1,size=d*n_test).reshape([-1,d])

    if model_type == "friedman":
        true = (10 * np.sin(np.pi*X_test[:,0]*X_test[:,1]) + 20*(X_test[:,2]-0.5)**2 + 10 * X_test[:,3] + 5 * X_test[:,4]).reshape([-1,1])
    elif model_type == "linear":
        true = (np.sum(X_test[:,:4],1)).reshape([-1,1])
    elif model_type == "constant":
        true = (np.ones(N)*2).reshape([-1,1])

    return X, Y, X_test, true

def predict_var(self,X_test,X, Y,lh_bias,deriv_bias):
    # prediction on the testing dataset
    predict_all = np.zeros((X_test.shape[0], self.n_estimators))
    for t_idx in range(self.n_estimators):
        predict_all[:, t_idx] = self.estimators_[t_idx].predict(X_test)
    pred = np.mean(predict_all, axis = 1)

    inbag_times_ = self.inbag_times_

    m = X_test.shape[0]
    
    # infinetesimal jacknnife and its correction
    f_centered = predict_all - np.mean(predict_all, axis=1).reshape(m, 1)
    i_centered = inbag_times_ - np.mean(inbag_times_, axis=1).reshape(self.n_samples_, 1)
    corr = np.dot(f_centered, i_centered.T) / self.n_estimators
    cov = np.dot(corr, corr.T)
    zetan_full = np.cov(predict_all)
    cov_ij = cov + zetan_full / self.n_estimators
    cov_ijc = cov + (1-np.sum(np.diag(np.cov(inbag_times_))))*zetan_full / self.n_estimators

    # deriv for IJ without correction
    U_ij = corr * corr.shape[1]
    
    # V-stat correction
    cond_exp_full = np.zeros((self.n_samples_, m))

    for i in range(self.n_samples_):
        cond_exp_full[i, :] = np.average(predict_all, weights = inbag_times_[i, :], axis = 1)

    inbag_times_ = inbag_times_[np.sum(inbag_times_, axis = 1) > 0, ]

    nk = np.sum(inbag_times_, axis = 1)
    K = len(nk)
    C = np.sum(nk)     

    SSr = np.matmul((cond_exp_full - pred).T*nk,cond_exp_full - pred)

    SSe = [0] * m

    for i in range(K):
        diff = predict_all[:, inbag_times_[i, :]>=1].T - cond_exp_full[i, :]
        SSei = np.matmul(diff.T,diff)
        SSe += SSei

    sigma_e_squared = SSe / (C - K)

    sigma_M_squared = (SSr - (K - 1) * sigma_e_squared) / (C - np.sum(nk ** 2) / C)
    
    sigma_M_squared_without = SSr / C

    cov_c = (self.n_subsamples_ ** 2) / self.n_samples_ * sigma_M_squared

    # Calculate deriv for bias correction estimator
    if lh_bias == True:
        predict_all_train = np.zeros((X.shape[0], self.n_estimators))
        for t_idx in range(self.n_estimators):
            predict_all_train[:, t_idx] = self.estimators_[t_idx].predict(X)
        pred_train = np.mean(predict_all_train, axis = 1)

        inbag_times_ = self.inbag_times_
        train_nodes = self.apply(X)
        test_nodes = self.apply(X_test)
        Dvec = Y.flatten() - pred_train
        Cmat = np.zeros([X_test.shape[0],X.shape[0]])
        for i in range(X_test.shape[0]):
            for j in range(X.shape[0]):
                Cmat[i,j] = np.sum((test_nodes[i,:] == train_nodes[j,:]) * (inbag_times_[j,:] == 0))
        r = np.matmul(Cmat,np.diag(Dvec))
        p = np.sum(r,1).reshape([-1,1])
        q = np.sum(Cmat,1).reshape([-1,1])
        bias = p/q
        ddmat = (r - Cmat*bias)/q
        U_bias = Cmat.shape[1] * ddmat
        cov_bias = np.matmul(U_bias,U_bias.T)/(Cmat.shape[1]**2)  

        if deriv_bias:
            return cov_ij,cov_ijc, cov_c, cov_bias, U_ij,U_bias, pred.reshape([-1,1]), bias
        else:
            return cov_ij,cov_ijc, cov_c, cov_bias, U_ij, pred.reshape([-1,1]), bias

    else:
        predict_all_train = np.zeros((X.shape[0], self.n_estimators))
        for t_idx in range(self.n_estimators):
            predict_all_train[:, t_idx] = self.estimators_[t_idx].predict(X)
        pred_train = np.mean(predict_all_train, axis = 1)
        return cov_ij,cov_ijc, cov_c, U_ij, pred.reshape([-1,1]), pred_train.reshape([-1,1])

def cal_dir_deriv_XGB(X,Y,X_test,n_estimators,n_sub): 
    N = X.shape[0]
    predict_all = np.zeros((X_test.shape[0], n_estimators))
    inbag_times_ = np.zeros((X.shape[0], n_estimators))
    for t_idx in range(n_estimators):
        select = np.random.choice(np.arange(N),n_sub,replace=True)
        X_boot = X[select,:]
        Y_boot = Y[select,:]
        xgb_r = xg.XGBRegressor(objective ='reg:squarederror')
        xgb_r.fit(X_boot, Y_boot.flatten())
        predict_all[:, t_idx] =  xgb_r.predict(X_test)
        for index in select:
            inbag_times_[index,t_idx] += 1
    pred = np.mean(predict_all, axis = 1)
    
    m = X_test.shape[0]

    # infinetesimal jacknnife and its correction
    f_centered = predict_all - np.mean(predict_all, axis=1).reshape(m, 1)
    i_centered = inbag_times_ - np.mean(inbag_times_, axis=1).reshape(N, 1)
    corr = np.dot(f_centered, i_centered.T) / n_estimators
    cov = np.dot(corr, corr.T)
    zetan_full = np.cov(predict_all)
    cov_ij = cov + zetan_full / n_estimators
    cov_ijc = cov + (1-np.sum(np.diag(np.cov(inbag_times_))))*zetan_full / n_estimators

    # deriv for IJ without correction
    U_ij = corr * corr.shape[1]
    
    # V-stat correction
    cond_exp_full = np.zeros((N, m))

    for i in range(N):
        cond_exp_full[i, :] = np.average(predict_all, weights = inbag_times_[i, :], axis = 1)

    inbag_times_ = inbag_times_[np.sum(inbag_times_, axis = 1) > 0, ]

    nk = np.sum(inbag_times_, axis = 1)
    K = len(nk)
    C = np.sum(nk)     

    SSr = np.matmul((cond_exp_full - pred).T*nk,cond_exp_full - pred)

    SSe = [0] * m

    for i in range(K):
        diff = predict_all[:, inbag_times_[i, :]>=1].T - cond_exp_full[i, :]
        SSei = np.matmul(diff.T,diff)
        SSe += SSei

    sigma_e_squared = SSe / (C - K)

    sigma_M_squared = (SSr - (K - 1) * sigma_e_squared) / (C - np.sum(nk ** 2) / C)
    
    sigma_M_squared_without = SSr / C

    cov_c = (n_sub ** 2) / N * sigma_M_squared

    return cov_ij,cov_ijc, cov_c, U_ij, pred.reshape([-1,1])

def cal_dir_deriv_RF(X,Y,X_test,n_estimators,n_sub,max_depth,lh_bias,deriv_bias): 
    if max_depth == "full":
        model = RandomForestRegressor(n_estimators = n_estimators)
    else:
        model = RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth)
    model.fit(X, Y.ravel(), n_subsamples = n_sub,replace = True)
    result = predict_var(model, X_test,X,Y,lh_bias,deriv_bias)
    return result

def LM_pred(X,X_test,Y,N):
    n_test = X_test.shape[0]
    X0 = np.concatenate([X,np.ones(N).reshape([-1,1])],axis=1)
    X_test0 = np.concatenate([X_test,np.ones(n_test).reshape([-1,1])],axis=1)
    XtX = np.matmul(X0.T,X0)
    try:
        theta = np.matmul(np.linalg.inv(XtX),np.matmul(X0.T,Y))
    except:
        theta = np.matmul(np.linalg.pinv(XtX),np.matmul(X0.T,Y))
    D1 = X_test0
    H = 2*XtX/N
    D2 = 2 * X0 * (Y - np.matmul(X0,theta))
    try:
        U = np.matmul(np.matmul(D1,np.linalg.inv(H)),D2.T)
    except:
        U = np.matmul(np.matmul(D1,np.linalg.pinv(H)),D2.T)
    Sigma = np.matmul(U,U.T)/(N**2)
    pred_train = np.matmul(X0,theta)
    pred_test = np.matmul(X_test0,theta)
    return Sigma, U, pred_train,pred_test

def predict_var_real(self,X_test,X_test_all,X, Y,lh_bias,deriv_bias):
    # prediction on the testing dataset
    predict_all = np.zeros((X_test.shape[0], self.n_estimators))
    for t_idx in range(self.n_estimators):
        predict_all[:, t_idx] = self.estimators_[t_idx].predict(X_test)
    pred = np.mean(predict_all, axis = 1)

    inbag_times_ = self.inbag_times_

    m = X_test.shape[0]
    
    # infinetesimal jacknnife and its correction
    f_centered = predict_all - np.mean(predict_all, axis=1).reshape(m, 1)
    i_centered = inbag_times_ - np.mean(inbag_times_, axis=1).reshape(self.n_samples_, 1)
    corr = np.dot(f_centered, i_centered.T) / self.n_estimators
    cov = np.dot(corr, corr.T)
    zetan_full = np.cov(predict_all)
    cov_ij = cov + zetan_full / self.n_estimators
    cov_ijc = cov + (1-np.sum(np.diag(np.cov(inbag_times_))))*zetan_full / self.n_estimators

    # deriv for IJ without correction
    U_ij = corr * corr.shape[1]
    
    # V-stat correction
    cond_exp_full = np.zeros((self.n_samples_, m))

    for i in range(self.n_samples_):
        cond_exp_full[i, :] = np.average(predict_all, weights = inbag_times_[i, :], axis = 1)

    inbag_times_ = inbag_times_[np.sum(inbag_times_, axis = 1) > 0, ]

    nk = np.sum(inbag_times_, axis = 1)
    K = len(nk)
    C = np.sum(nk)     

    SSr = np.matmul((cond_exp_full - pred).T*nk,cond_exp_full - pred)

    SSe = [0] * m

    for i in range(K):
        diff = predict_all[:, inbag_times_[i, :]>=1].T - cond_exp_full[i, :]
        SSei = np.matmul(diff.T,diff)
        SSe += SSei

    sigma_e_squared = SSe / (C - K)

    sigma_M_squared = (SSr - (K - 1) * sigma_e_squared) / (C - np.sum(nk ** 2) / C)
    
    sigma_M_squared_without = SSr / C

    cov_c = (self.n_subsamples_ ** 2) / self.n_samples_ * sigma_M_squared

    # Calculate deriv for bias correction estimator
    if lh_bias == True:
        predict_all_train = np.zeros((X.shape[0], self.n_estimators))
        for t_idx in range(self.n_estimators):
            predict_all_train[:, t_idx] = self.estimators_[t_idx].predict(X)
        pred_train = np.mean(predict_all_train, axis = 1)

        inbag_times_ = self.inbag_times_
        train_nodes = self.apply(X)


        test_nodes = self.apply(X_test)
        Dvec = Y.flatten() - pred_train
        Cmat = np.zeros([X_test.shape[0],X.shape[0]])
        for i in range(X_test.shape[0]):
            for j in range(X.shape[0]):
                Cmat[i,j] = np.sum((test_nodes[i,:] == train_nodes[j,:]) * (inbag_times_[j,:] == 0))
        r = np.matmul(Cmat,np.diag(Dvec))
        p = np.sum(r,1).reshape([-1,1])
        q = np.sum(Cmat,1).reshape([-1,1])
        bias = p/q
        ddmat = (r - Cmat*bias)/q
        U_bias = Cmat.shape[1] * ddmat
        cov_bias = np.matmul(U_bias,U_bias.T)/(Cmat.shape[1]**2)  

        test_nodes = self.apply(X_test_all)
        Dvec = Y.flatten() - pred_train
        Cmat = np.zeros([X_test_all.shape[0],X.shape[0]])
        for i in range(X_test_all.shape[0]):
            for j in range(X.shape[0]):
                Cmat[i,j] = np.sum((test_nodes[i,:] == train_nodes[j,:]) * (inbag_times_[j,:] == 0))
        r = np.matmul(Cmat,np.diag(Dvec))
        p = np.sum(r,1).reshape([-1,1])
        q = np.sum(Cmat,1).reshape([-1,1])
        bias_all = p/q

        if deriv_bias:
            return cov_ij,cov_ijc, cov_c, cov_bias, U_ij,U_bias, pred.reshape([-1,1]), bias, bias_all
        else:
            return cov_ij,cov_ijc, cov_c, cov_bias, U_ij, pred.reshape([-1,1]), bias, bias_all

    else:
        predict_all_train = np.zeros((X.shape[0], self.n_estimators))
        for t_idx in range(self.n_estimators):
            predict_all_train[:, t_idx] = self.estimators_[t_idx].predict(X)
        pred_train = np.mean(predict_all_train, axis = 1)
        return cov_ij,cov_ijc, cov_c, U_ij, pred.reshape([-1,1]), pred_train.reshape([-1,1])

def cal_dir_deriv_RF_real(X,Y,X_test,X_test_all,n_estimators,n_sub,max_depth,lh_bias,deriv_bias): 
    if max_depth == "full":
        model = RandomForestRegressor(n_estimators = n_estimators)
    else:
        model = RandomForestRegressor(n_estimators = n_estimators, max_depth=max_depth)
    model.fit(X, Y.ravel(), n_subsamples = n_sub,replace = True)
    if deriv_bias==True:
        result = predict_var_real(model, X_test, X_test_all,X,Y,lh_bias,deriv_bias)
    else:
        result = predict_var(model, X_test,X,Y,lh_bias,deriv_bias)
    pred_test_all = model.predict(X_test_all)
    return result, pred_test_all

def LM_pred_real(X,X_test,X_test_all,Y,N):
    n_test = X_test.shape[0]
    X0 = np.concatenate([X,np.ones(N).reshape([-1,1])],axis=1)
    X_test0 = np.concatenate([X_test,np.ones(n_test).reshape([-1,1])],axis=1)
    X_test_all0 = np.concatenate([X_test_all,np.ones(X_test_all.shape[0]).reshape([-1,1])],axis=1)
    XtX = np.matmul(X0.T,X0)
    theta = np.matmul(np.linalg.pinv(XtX),np.matmul(X0.T,Y))
    D1 = X_test0
    H = 2*XtX/N
    D2 = 2 * X0 * (Y - np.matmul(X0,theta))
    U = np.matmul(np.matmul(D1,np.linalg.pinv(H)),D2.T)
    Sigma = np.matmul(U,U.T)/(N**2)
    pred_train = np.matmul(X0,theta)
    pred_test = np.matmul(X_test0,theta)
    pred_test_all = np.matmul(X_test_all0,theta)
    return Sigma, U, pred_train,pred_test,pred_test_all

def cal_dir_deriv_XGB_real(X,Y,X_test,X_test_all,n_estimators,n_sub): 
    N = X.shape[0]
    predict_all = np.zeros((X_test.shape[0], n_estimators))
    predict_test_all = np.zeros((X_test_all.shape[0], n_estimators))
    inbag_times_ = np.zeros((X.shape[0], n_estimators))

    for t_idx in range(n_estimators):
        select = np.random.choice(np.arange(N),n_sub,replace=True)
        X_boot = X[select,:]
        Y_boot = Y[select,:]
        xgb_r = xg.XGBRegressor(objective ='reg:squarederror')

        xgb_r.fit(X_boot, Y_boot.flatten())
        predict_all[:, t_idx] =  xgb_r.predict(X_test)

        predict_test_all[:, t_idx] =  xgb_r.predict(X_test_all)  

        for index in select:
            inbag_times_[index,t_idx] += 1
    pred = np.mean(predict_all, axis = 1)
    pred_test_all = np.mean(predict_test_all, axis = 1)
    
    m = X_test.shape[0]

    # infinetesimal jacknnife and its correction
    f_centered = predict_all - np.mean(predict_all, axis=1).reshape(m, 1)
    i_centered = inbag_times_ - np.mean(inbag_times_, axis=1).reshape(N, 1)
    corr = np.dot(f_centered, i_centered.T) / n_estimators
    cov = np.dot(corr, corr.T)
    zetan_full = np.cov(predict_all)
    cov_ij = cov + zetan_full / n_estimators
    cov_ijc = cov + (1-np.sum(np.diag(np.cov(inbag_times_))))*zetan_full / n_estimators

    # deriv for IJ without correction
    U_ij = corr * corr.shape[1]
    
    # V-stat correction
    cond_exp_full = np.zeros((N, m))

    for i in range(N):
        cond_exp_full[i, :] = np.average(predict_all, weights = inbag_times_[i, :], axis = 1)

    inbag_times_ = inbag_times_[np.sum(inbag_times_, axis = 1) > 0, ]

    nk = np.sum(inbag_times_, axis = 1)
    K = len(nk)
    C = np.sum(nk)     

    SSr = np.matmul((cond_exp_full - pred).T*nk,cond_exp_full - pred)

    SSe = [0] * m

    for i in range(K):
        diff = predict_all[:, inbag_times_[i, :]>=1].T - cond_exp_full[i, :]
        SSei = np.matmul(diff.T,diff)
        SSe += SSei

    sigma_e_squared = SSe / (C - K)

    sigma_M_squared = (SSr - (K - 1) * sigma_e_squared) / (C - np.sum(nk ** 2) / C)
    
    sigma_M_squared_without = SSr / C

    cov_c = (n_sub ** 2) / N * sigma_M_squared
    result = cov_ij,cov_ijc, cov_c, U_ij, pred.reshape([-1,1])
    return result,pred_test_all

def get_real(path):
    df0 = pd.read_csv(path, encoding = "ISO-8859-1")
    df = df0.copy()
    df = df.drop(columns=['id','url','Cid','price'])

    def extract_year(x):
        return x[0:4]
    df['tradeTime'] = df['tradeTime'].apply(extract_year)

    df['tradeTime'] = pd.to_numeric(df['tradeTime'])
    df['livingRoom'] = df['livingRoom'].apply(pd.to_numeric, errors='coerce')
    df['drawingRoom'] = df['drawingRoom'].apply(pd.to_numeric, errors='coerce')
    df['bathRoom'] = df['bathRoom'].apply(pd.to_numeric, errors='coerce')
    df['constructionTime'] = df['constructionTime'].apply(pd.to_numeric, errors='coerce')

    def floorHeight(x):
        try:
            return int(x.split(' ')[1])
        except:
            return np.nan
        
    df['floorHeight'] = df['floor'].apply(floorHeight)

    df = df.drop(columns=['floor'])

    cols_to_get_dummies = ['buildingType','renovationCondition','buildingStructure','district']
    df = pd.get_dummies(data=df, columns=cols_to_get_dummies,drop_first=True)
    df = df.dropna()

    X = np.float64(np.array(df.drop(columns=['totalPrice'])))
    y = np.log(np.float64(np.array(df['totalPrice'])))

    X_s, y_s = shuffle(X, y, random_state=2)

    size = X_s.shape[0]
    num_train = 150000
    X_train = X_s[:num_train] 
    X_test_all = X_s[num_train:] 
    y_train = y_s[:num_train]
    y_test = y_s[num_train:]

    X_train_split = np.array_split(X_train, 50)
    y_train_split = np.array_split(y_train, 50)

    np.random.seed(2)
    select = np.sort(np.random.choice(range(len(y_test)),100,replace=False))
    X_test = X_test_all[select,:]

    return X_train_split, y_train_split, X_test_all, y_test, X_test