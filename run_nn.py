from nn_jackknife_utils import*

count = 0
for N in [5000,1000]:
    for act in ['relu','sigmoid']:
        for fix_s in [False,True]:            
            for model_type in ['linear','friedman','constant']:
                count += 1
                string = str(count)
                for h_dim in [1,3,5,10,20]:
                    d = 6
                    n_test = 100
                    EPOCH = 1000
                    lr = 0.01
                    tune = True
                    param_dict = {'N':N,'d':d,'n_test':n_test,
                                  'h_dim':h_dim,'activation':act,'learning_rate':lr,'model_type':model_type,'epoch':EPOCH}
                    print(param_dict)

                    epoch_choice = [1000]
                    lam_choice = [1e-03]
                    cond_shred_choice = [100000]
                    n_iter = 200


                    cov_mat_all = []
                    y_pred_all = []
                    param_all=[]
                    for seed in tqdm(range(n_iter)):
                        y_pred_ls, cov_mat_ls, param_ls, true = simulate_NN(seed,N,d,n_test,h_dim,EPOCH,lr,act,model_type,fix_s,epoch_choice,lam_choice,cond_shred_choice)
                        cov_mat_all.append(cov_mat_ls)
                        y_pred_all.append(y_pred_ls)
                        param_all.append(param_ls)

                    for k in range(np.int(len(epoch_choice)*len(lam_choice)*len(cond_shred_choice))):
                        cov_mat_ls = [item[k] for item in cov_mat_all]
                        y_pred_ls = [item[k] for item in y_pred_all]
                        param_ls = [item[k] for item in param_all]
                        param_dict['epoch']= param_all[0][k][0]
                        param_dict['lam']= param_all[0][k][1]
                        param_dict['cond_shred']= param_all[0][k][2] 

                        monto_carlo_var = []
                        est_var = []
                        coverage = []
                        coverage_debiased = []
                        est_mean = []
                        est_var_detail = []
                        for i in range(len(y_pred_ls[0])):
                            monto_carlo_var.append(np.var(np.array([y[i] for y in y_pred_ls])))
                            est_var.append(np.mean([cov[i,i] for cov in cov_mat_ls]))
                            est_var_detail.append([cov[i,i] for cov in cov_mat_ls])
                            est_mean.append(np.mean([y[i] for y in y_pred_ls]))
                            upper = np.array([y[i] for y in y_pred_ls]).flatten() + 1.96*np.sqrt(np.array([cov[i,i] for cov in cov_mat_ls]))
                            lower = np.array([y[i] for y in y_pred_ls]).flatten() - 1.96*np.sqrt(np.array([cov[i,i] for cov in cov_mat_ls]))
                            true_mean = np.mean([y[i] for y in y_pred_ls])
                            coverage.append(np.mean((true[i]<upper)*(true[i]>lower)))
                            coverage_debiased.append(np.mean((true_mean<upper)*(true_mean>lower)))
                        df = pd.DataFrame(data={'Truth':true,'Estimate':est_mean,'MC_var': monto_carlo_var, 'Est_var': est_var,'Coverage': coverage,'Coverage_Debiased': coverage_debiased})
                        print(param_dict)
                        print(df)

                        df.to_csv('data/'+string+'_param'+str(k)+'_hdim_'+str(h_dim)+'.csv')
                        with open('data/'+string+'_param'+str(k)+'_hdim_'+str(h_dim)+'.pkl', 'wb') as f:
                            pickle.dump(param_dict, f)
                        np.save('data/var_'+string+'_param'+str(k)+'_hdim_'+str(h_dim),est_var_detail)
                        np.save('data/true_'+string+'_param'+str(k)+'_hdim_'+str(h_dim),true)

                        open_file = open('data/var_'+string+'_param'+str(k)+'_hdim_'+str(h_dim)+".pkl", "wb")
                        pickle.dump(y_pred_all, open_file)
                        open_file.close()
                    
df_all = []
for i in range(24):
    print("Result for "+str(i+1))
    df_ls = []
    for h_dim in [1,3,5,10,20]:
        file = 'data/var_'+str(i+1)+'_param0'+'_hdim_'+str(h_dim)
        with open(file+'.pkl', 'rb') as f:
                result_all = pickle.load(f)
        file = 'data/'+str(i+1)+'_param0'+'_hdim_'+str(h_dim)
        with open(file+'.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)
        df_load = pd.read_csv(file+".csv", index_col=0)
        true = np.array(df_load['Truth']).reshape([1,-1])
        pred = np.array(result_all)[:,0,:,0]
        MSE = np.mean((pred - true)**2,0)
        df_load['MSE'] = MSE
        df_ls.append(df_load)    
    df_all.append(df_ls)        
open_file = open("data/df_info.pkl", "wb")
pickle.dump(df_all, open_file)
open_file.close()