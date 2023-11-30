import numpy as np
import openml
import warnings
warnings.simplefilter("ignore", UserWarning)
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os
import sys
sys.path.insert(0, '..')
from benchmarks.Group_MulltiFoldBenchmark import Group_MultiFold_Space,XGB_NAME,LINEAR_SVM_NAME,DT_NAME,RF_NAME,RBF_SVM_NAME
from global_utilities.global_util import csv_postfix,parse_directory
from pathlib import Path
import numpy as np

from BayesianOptimizers.Conditional_BayesianOptimization.smac_hpo import SMAC_HPO
from BayesianOptimizers.Conditional_BayesianOptimization.Group_Random_Search import Group_Random_Search
from BayesianOptimizers.Experimental.Pavlos_BO import Pavlos_BO
from BayesianOptimizers.Experimental.PavlosV2 import PavlosV2
from BayesianOptimizers.Conditional_BayesianOptimization.MultiFold_Group_Smac_base import MultiFold_Group_Bayesian_Optimization
from BayesianOptimizers.Conditional_BayesianOptimization.Progressive_group_smac import Progressive_BO
from BayesianOptimizers.Conditional_BayesianOptimization.progressive_wholebatch_bo import Batch_Progressive_BO
from BayesianOptimizers.Conditional_BayesianOptimization.progressive_smallbatch_bo import miniBatch_Progressive_BO
from BayesianOptimizers.Conditional_BayesianOptimization.Switch_Surrogate_BO import Switch_BO
from BayesianOptimizers.Conditional_BayesianOptimization.greedy_smac_group import Greedy_SMAC
from BayesianOptimizers.Conditional_BayesianOptimization.Group_SMAC_base import Group_Bayesian_Optimization
from BayesianOptimizers.Conditional_BayesianOptimization.Mango import Mango
from BayesianOptimizers.Conditional_BayesianOptimization.Optuna import Optuna
from BayesianOptimizers.Conditional_BayesianOptimization.hyperopt import HyperOpt
from BayesianOptimizers.Conditional_BayesianOptimization.smac_instance_hpo import SMAC_Instance_HPO
from BayesianOptimizers.Conditional_BayesianOptimization.hyperband import HyperBand
from BayesianOptimizers.Conditional_BayesianOptimization.RF_BO_without_config_norm import Bayesian_Optimization_without_Norm


from csv import writer
import time 


def run_benchmark_total(optimizers_used =[],bench_config={},save=True):
    assert optimizers_used != []
    assert bench_config != {}

    #Optimizer related
    n_init,max_evals = bench_config['n_init'],bench_config['max_evals']

    #Dataset related
    data_ids ,n_seeds = bench_config['data_ids'],bench_config['n_seeds']

    data_repo  = bench_config['data_repo']

    #Benchmark related fields
    type_of_bench = bench_config['type_of_bench'] 
    benchmark_name = bench_config['bench_name']
    benchmark_class = bench_config['bench_class']
    
    optimizers_list = optimizers_used

    assert optimizers_list != [] or optimizers_list != None


    main_directory = os.getcwd()
    
    
    for task_id in data_ids:
        task_id_str = 'Dataset' +str(task_id)
        for seed in n_seeds:
            Seed_id_str = 'Seed' + str(seed)
            for opt in optimizers_list: 
            
                score_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Metric',data_repo,task_id_str,Seed_id_str,opt)
                total_time_per_optimizer_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Total_Time',data_repo,task_id_str,Seed_id_str,opt)
                config_per_group_directory = os.path.join(os.getcwd(),main_directory,type_of_bench,benchmark_name,'Configurations',data_repo,task_id_str,Seed_id_str,opt)

                benchmark_ = benchmark_class(task_id=task_id,rng=seed,data_repo=data_repo)
                #Get the config Space
                configspace,config_dict = benchmark_.get_configuration_space()

                #Get the benchmark.
                smac_objective_function = benchmark_.smac_objective_function
                
                #Get the objective_function per fold.
                smac_objective_function_per_fold = benchmark_.smac_objective_function_per_fold

                #Get the benchmark.
                objective_function = benchmark_.objective_function
                
                #Get the objective_function per fold.
                objective_function_per_fold = benchmark_.objective_function_per_fold


                optuna_objective = benchmark_.optuna_objective
                
                hyperopt_objective = benchmark_.hyperopt_objective_function
                hyperopt_space = benchmark_.hyper_opt_space()


                benchmark_multifidelity = benchmark_class(task_id=task_id,rng=seed,data_repo=data_repo,is_multifidelity=True)
                multifidelity_objective = benchmark_multifidelity.objective_function_multifidelity
                multifidelity_configspace,multifidelity_config_dict = benchmark_multifidelity.get_configuration_space_multifidelity()




                mango_config_space = benchmark_.get_mango_config_space()
                mango_objectives  ={ DT_NAME: benchmark_.mango_objective_dt,
                                     XGB_NAME: benchmark_.mango_objective_xgb,
                                     LINEAR_SVM_NAME : benchmark_.mango_objective_LinearSVM,
                                     RBF_SVM_NAME : benchmark_.mango_objective_RBFSVM,
                                     RF_NAME: benchmark_.mango_objective_RF}

                print('Currently running ' + opt + ' on seed ' + str(seed) + ' dataset ' + str(task_id) )

                
                if opt == 'Random_Search':
                    Optimization = Group_Random_Search(f=objective_function,configuration_space= configspace,n_init = n_init,max_evals= max_evals,random_seed=seed)
                elif opt == 'Pavlos':
                    Optimization = Pavlos_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'Multi_RF_Local':
                    Optimization = MultiFold_Group_Bayesian_Optimization(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'PavlosV2':
                    Optimization = PavlosV2(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'Progressive_BO':
                    Optimization = Progressive_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'Switch_BO':
                    Optimization = Switch_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'Greedy_SM':
                    Optimization = Greedy_SMAC(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'SMAC':
                    Optimization = SMAC_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                    repo=data_repo,max_evals=max_evals,seed=seed,objective_function=smac_objective_function,n_workers=1)
                elif opt == 'SMAC_Instance':
                    Optimization= SMAC_Instance_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                             repo=repo,max_evals=5*max_evals,seed=seed,objective_function=smac_objective_function_per_fold)
                elif opt == 'SMAC2':
                    Optimization = SMAC_HPO(configspace=configspace,config_dict=config_dict,task_id=task_id,
                    repo=data_repo,max_evals=max_evals,seed=seed,objective_function=smac_objective_function,n_workers=1,init_evals = 5*n_init)
                elif opt =='RF_Local':
                    Optimization =Group_Bayesian_Optimization(f=objective_function, model='RF' ,lb= None, ub =None ,configuration_space=config_dict,\
                                                n_init=n_init,max_evals=max_evals,initial_design=None,random_seed=seed,maximizer='Sobol_Local')
                elif opt =='RF_Local_extensive':
                    Optimization =Group_Bayesian_Optimization(f=objective_function, model='RF' ,lb= None, ub =None ,configuration_space=config_dict,\
                                                n_init=n_init,max_evals=max_evals,initial_design=None,random_seed=seed,maximizer='Sobol_Local',extensive='Hyper')
                elif opt =='RF_Local_No_STD':
                    Optimization =Group_Bayesian_Optimization(f=objective_function, model='RF' ,lb= None, ub =None ,configuration_space=config_dict,\
                                                n_init=n_init,max_evals=max_evals,initial_design=None,random_seed=seed,maximizer='Sobol_Local',std_out='Yes')
                elif opt == 'Mango2':
                    Optimization = Mango(mango_config_space,n_init,max_evals,seed,mango_objectives)
                elif opt == 'Optuna':
                    Optimization = Optuna(max_evals,seed,optuna_objective)
                elif opt == 'HyperOpt':
                    Optimization = HyperOpt(max_evals,seed,hyperopt_objective,hyperopt_space)
                elif opt == 'Hyperband':
                    Optimization = HyperBand(configspace=multifidelity_configspace,config_dict=multifidelity_config_dict,task_id=task_id,
                    repo=data_repo,max_evals=max_evals,seed=seed,objective_function=multifidelity_objective,n_workers=1)
                elif opt == 'Progr_Batch_BO':
                    Optimization = Batch_Progressive_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'MiniBatch_Progressive':
                    Optimization = miniBatch_Progressive_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5)
                elif opt == 'RF_Local_std0.2':
                    Optimization =Group_Bayesian_Optimization(f=objective_function, model='RF' ,lb= None, ub =None ,configuration_space=config_dict,\
                                                n_init=n_init,max_evals=max_evals,initial_design=None,random_seed=seed,maximizer='Sobol_Local',stdev=0.2)
                elif opt == 'Progressive_BO_std0.2':
                    Optimization = Progressive_BO(f=objective_function_per_fold, model='RF' ,lb= None, ub =None , configuration_space= config_dict ,\
                    initial_design=None,n_init = n_init, max_evals = max_evals, batch_size=1 ,verbose=True,random_seed=seed,maximizer = 'Sobol_Local',n_folds=5,stdev=0.2)
                elif opt == 'Experimental':
                    Optimization = Bayesian_Optimization_without_Norm(f=objective_function, model='RF' ,lb= None, ub =None ,configuration_space=config_dict,\
                                                n_init=n_init,max_evals=max_evals,initial_design=None,random_seed=seed,maximizer='Sobol_Local')
                
                else: 
                    print(opt)
                    raise RuntimeError
                
                          
                """
                """
                
                
                
                
                start_time = time.time()
                Optimization.run()
                m_time = time.time()-start_time
                print('Measured Total Time ',m_time)
                
                
                #Change this.
                y_evaluations = Optimization.fX
                total_time_evaluations = Optimization.total_time

                if save == True:
                    try:
                        Path(score_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(total_time_per_optimizer_directory).mkdir(parents=True, exist_ok=True)
                        Path(config_per_group_directory).mkdir(parents=True, exist_ok=True)
                    except FileExistsError:
                        print("Folder is already there")
                    else:
                        print("Folder is created there")
                        
                    pd.DataFrame(y_evaluations).to_csv( parse_directory([ score_per_optimizer_directory, opt+csv_postfix ]))
                    pd.DataFrame(total_time_evaluations).to_csv( parse_directory([ total_time_per_optimizer_directory, opt+csv_postfix ]))

                    if 'SMAC' in opt or 'ROAR' in opt or 'Hyperband' in opt:
                        #Save configurations and y results for each group.
                        for group in Optimization.save_configuration:
                            Optimization.save_configuration[group].to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    elif opt == 'Multi_RF_Local' or opt == 'MiniBatch_Progressive' or opt == 'Progr_Batch_BO' or opt == 'Progressive_BO' or opt =='Switch_BO' or opt =='GreedySM' or opt == 'RF_Local' or opt =='RF_Local_extensive' or opt == 'RF_Local_No_STD':
                        for group in Optimization.object_per_group:
                            X_df = Optimization.object_per_group[group].X_df
                            y_df = pd.DataFrame({'y':Optimization.object_per_group[group].fX})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    elif opt =='Random_Search':
                        #Save configurations and y results for each group.
                        for group in Optimization.fX_per_group:
                            X_df = Optimization.X_per_group[group]
                            y_df = pd.DataFrame({'y':Optimization.fX_per_group[group]})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                    elif opt == 'Pavlos' or opt == 'PavlosV2':
                        for group in Optimization.object_per_group:
                            X_df = Optimization.object_per_group[group].X_df
                            y_df = pd.DataFrame({'y':Optimization.object_per_group[group].fX})
                            pd.concat([X_df,y_df],axis=1).to_csv( parse_directory([ config_per_group_directory, group+csv_postfix ]))
                        pd.DataFrame({'GroupName':Optimization.X_group}).to_csv( parse_directory([ config_per_group_directory, 'group_index'+csv_postfix ]))
                    
    






def get_openml_data(speed = None):
    # 2074 needs 15 hours for 3 seeds per optimizer.
    assert speed !=None
    if speed == 'fast':
        return [14954,11,3918,3917,3021,43,167141,9952,] #[]
    return [2074, 9976,10101] #[ 2074, 9976,10101] #,9910,167125, 9910,167125

def get_jad_data(speed = None):        
    assert speed !=None
    if speed == 'fast':
        return [839,842,851,850,1114,847]
    #  on all seeds 
    return  [843,883] #[843,883] 866

def get_automl_data(speed=None):
    assert speed !=None
    if speed == 'fast':
        return [12,146822,168911,3,168912] #146818,10101,53,31,9981,146821,
    return [7592,146606]

if __name__ == '__main__':
    config_of_data = { 'Jad':{'data_ids':get_jad_data},
                    'OpenML': {'data_ids':get_openml_data} ,
                    'AutoML' : {'data_ids':get_automl_data}     }
    
    #9976 datasets hasn't run for SMAC.
    # 'SMAC','Random_Search','Progressive_BO' 
    # 'RF_Local','SMAC','Random_Search','Multi_RF_Local','Progressive_BO','Greedy_SM'
    opt_list = ['Experimental'] #['Switch_BO','SMAC','Random_Search','Multi_RF_Local','Progressive_BO'] # ,,,'RF_Local',] 'SMAC' 'SMAC_Instance' ,'SMAC' ,'Random_Search',, 'Pavlos','Random_Search','Multi_RF_Local' 'SMAC', 'Pavlos','Random_Search','Multi_RF_Local','Random_Search','Multi_RF_Local','Pavlos'
    for speed in ['fast','slow']: #
     # obtain the benchmark suite    
        for repo in ['OpenML','Jad']: #'
            #XGBoost Benchmark    
            xgb_bench_config =  {
                'n_init' : 10,
                'max_evals' : 1050,
                'n_datasets' : 1000,
                'data_ids' :  config_of_data[repo]['data_ids'](speed=speed),
                'n_seeds' : [1], 
                'type_of_bench': 'Main_Multi_Fold_Group_Space_Results',
                'bench_name' :'GROUP',
                'bench_class' : Group_MultiFold_Space,
                'data_repo' : repo
            } 
            run_benchmark_total(opt_list,xgb_bench_config)

