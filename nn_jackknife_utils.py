import numpy as np
import random
import keras
import pandas as pd
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense    
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model
import tensorflow as tf
import pickle
from numpy import linalg as LA
import copy
tf.keras.backend.set_floatx('float64')

class DNN(Model):
    def __init__(self,h_dim,act):
        super(DNN, self).__init__()
        self.d1 = Dense(h_dim, activation=act)
        self.d2 = Dense(1, activation=None)

    def call(self, x):
        h1 = self.d1(x)
        h2 = self.d2(h1)
        return h2

    def basis(self, x):
        h1 = self.d1(x)
        return h1

def cal_grad(X_tf,Y_tf,N,model,loss_object):
    shred = 1000
    if X_tf.shape[0] <= shred: 
        with tf.GradientTape() as tape:
            y_pred = model(X_tf)
            loss = (Y_tf - y_pred)*(Y_tf - y_pred)
        gradients = tape.jacobian(loss, model.trainable_variables)
        g_vec = np.concatenate([tf.reshape(g, [N, -1]) for g in gradients],axis=1)
        return g_vec
    else:
        n_rep = np.int(X_tf.shape[0]/shred)
        g_vec_ls = []
        for j in range(n_rep):
            with tf.GradientTape() as tape:
                y_pred = model(X_tf[(j*shred):((j+1)*shred),:])
                loss = (Y_tf[(j*shred):((j+1)*shred),:] - y_pred)*(Y_tf[(j*shred):((j+1)*shred),:] - y_pred)
            gradients = tape.jacobian(loss, model.trainable_variables)
            g_vec = np.concatenate([tf.reshape(g, [shred, -1]) for g in gradients],axis=1)      
            g_vec_ls.append(g_vec) 
        g_vec_arr = np.concatenate(g_vec_ls,0)
        return g_vec_arr

def cal_grad_test(x_tf,model,loss_object):
    with tf.GradientTape() as tape:
        y_pred = model(x_tf)
    gradients = tape.jacobian(y_pred, model.trainable_variables)
    g_vec = np.concatenate([tf.reshape(g, [x_tf.shape[0], -1]) for g in gradients],axis=1)
    return  g_vec   

def cal_jacobian(X_tf,Y_tf,h_dim,d,model,loss_object):
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y_pred = model(X_tf)
            loss = loss_object(Y_tf, y_pred)
        g = t1.gradient(loss, model.trainable_variables)
    h1 = t2.jacobian(g[0], model.trainable_variables)

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y_pred = model(X_tf)
            loss = loss_object(Y_tf, y_pred)
        g = t1.gradient(loss, model.trainable_variables)
    h2 = t2.jacobian(g[1], model.trainable_variables)

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y_pred = model(X_tf)
            loss = loss_object(Y_tf, y_pred)
        g = t1.gradient(loss, model.trainable_variables)
    h3 = t2.jacobian(g[2], model.trainable_variables)

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y_pred = model(X_tf)
            loss = loss_object(Y_tf, y_pred)
        g = t1.gradient(loss, model.trainable_variables)
    h4 = t2.jacobian(g[3], model.trainable_variables)

    n_params1 = h_dim*d
    n_params2 = h_dim
    n_params3 = h_dim
    n_params4 = 1

    h_mat_1 = np.concatenate([tf.reshape(h_i, [n_params1, -1]) for h_i in h1],axis=1)
    h_mat_2 = np.concatenate([tf.reshape(h_i, [n_params2, -1]) for h_i in h2],axis=1)
    h_mat_3 = np.concatenate([tf.reshape(h_i, [n_params3, -1]) for h_i in h3],axis=1)
    h_mat_4 = np.concatenate([tf.reshape(h_i, [n_params4, -1]) for h_i in h4],axis=1)
    h_mat = np.concatenate([h_mat_1,h_mat_2,h_mat_3,h_mat_4],axis=0)

    return h_mat

def simulate_NN(seed,N,d,n_test,h_dim,EPOCH,lr,act,model_type,fix_s,epoch_choice=None,lam_choice=None,cond_shred_choice=None,tune=True):
    np.random.seed(seed)
    X = np.random.uniform(-1,1,N*d).reshape([N,d])
    if model_type == "friedman":
        Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10 * X[:,3] + 5 * X[:,4] + np.random.normal(0,1,N)).reshape([-1,1])
    elif model_type == "linear":
        Y = (np.sum(X[:,:4],1)+ np.random.normal(0,1,N)).reshape([-1,1])
    elif model_type == "constant":
        Y = (np.ones(N)*2 + np.random.normal(0,1,N)).reshape([-1,1])

    X_tf = tf.convert_to_tensor(X)
    Y_tf = tf.convert_to_tensor(Y)

    np.random.seed(2021)
    X_test = np.random.uniform(-1,1,n_test*d).reshape([n_test,d])
    X_test_tf = tf.convert_to_tensor(X_test)   

    if model_type == "friedman":
        true = 10 * np.sin(np.pi*X_test[:,0]*X_test[:,1]) + 20*(X_test[:,2]-0.5)**2 + 10 * X_test[:,3] + 5 * X_test[:,4]
    elif model_type == "linear":
        true = np.sum(X_test[:,:4],1)
    elif model_type == "constant":
        true = np.ones(n_test)*2
    elif model_type == "mars":
        true = 10 * np.sin(np.pi*X_test[:,0]*X_test[:,1]) + 20*(X_test[:,2]-0.05)**2 + 10 * X_test[:,3] + 5 * X_test[:,4]

    # define optimization 
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    if tune == True:
        @tf.function
        def train_step(x_train, y_train):
            with tf.GradientTape() as tape:
                y_pred = model(x_train)
                loss = loss_object(y_train, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss,gradients
          
        # train the DNN model
        if fix_s == False:
            tf.random.set_seed(seed)
        else:
            tf.random.set_seed(0)
        # define the model object
        model = DNN(h_dim,act)

        cov_mat_candidate = []
        y_test_ls = []
        cov_mat_ls = []
        param_ls = []
        for k in range(EPOCH):
            loss,gradients = train_step(X_tf, Y_tf)
            if (k+1) in epoch_choice:
                g_train = cal_grad(X_tf,Y_tf,N,model,loss_object)
                h_mat0 = cal_jacobian(X_tf,Y_tf,h_dim,d,model,loss_object) 
                eigen_values, _ = LA.eig(h_mat0)  
                g_test = cal_grad_test(X_test_tf,model,loss_object)     
                y_test = model(X_test_tf)
                for lam in lam_choice:
                    for cond_shred in cond_shred_choice:
                        cond_num = np.max(np.abs(eigen_values))/np.min(np.abs(eigen_values))
                        if cond_num > cond_shred:
                            h_mat=h_mat0+np.diag(np.ones(h_mat0.shape[0]))*(lam * np.mean(eigen_values)- np.min(eigen_values))
                        else:
                            h_mat = h_mat0 
                        h_mat_inv = np.linalg.inv(h_mat)
                        U = np.matmul(-g_test,np.matmul(h_mat_inv,g_train.T))
                        cov_mat = np.matmul(U,U.T)/(N**2)   
                        y_test_ls.append(copy.deepcopy(y_test))   
                        cov_mat_ls.append(copy.deepcopy(cov_mat))  
                        param_ls.append([k+1,lam,cond_shred])

        return y_test_ls, cov_mat_ls, param_ls, true
    else:
        # train the DNN model
        tf.random.set_seed(seed)
        while True:
            @tf.function
            def train_step(x_train, y_train):
                with tf.GradientTape() as tape:
                    y_pred = model(x_train)
                    loss = loss_object(y_train, y_pred)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return loss,gradients

            model = DNN(h_dim,act)

            loss_ls = []
            for k in range(EPOCH):
                loss,gradients = train_step(X_tf, Y_tf)
                if k%1000 == 0:
                    loss_ls.append(loss)

            g_train = cal_grad(X_tf,Y_tf,N,model,loss_object)
            h_mat = cal_jacobian(X_tf,Y_tf,h_dim,d,model,loss_object)
            g_test = cal_grad_test(X_test_tf,model,loss_object)
            eigen_values, _ = LA.eig(h_mat)
            if np.sum(eigen_values==0)==0:
                break
        h_mat_inv = np.linalg.inv(h_mat)
        U = np.matmul(-g_test,np.matmul(h_mat_inv,g_train.T))
        cov_mat = np.matmul(U,U.T)/(N**2)

        y_test = model(X_test_tf)      

        return cov_mat,y_test,true