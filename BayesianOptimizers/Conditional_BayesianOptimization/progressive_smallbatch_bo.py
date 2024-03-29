import numpy as np
from BayesianOptimizers.Conditional_BayesianOptimization.Progressive_Batch_Group_EI import EI_Batch_per_group

import pandas as pd
import time



class miniBatch_Progressive_BO:
    """The Random Forest Based Regression Local Bayesian Optimization.udnn-cu11, minio, kiwisolver, Jinja2, importlib-metadata, emcee, Deprecated, autograd, alive-progress, xgboost, torch, stevedore, scikit-learn, requests-toolbelt, paramz, pandas, matplotlib, george, debtcollector, dask, ConfigSpace, click, autograd-gamma, torchvision, statsmodels, pymoo, oslo.utils, oslo.config, openml, gpytorch, GPy, formulaic, 
    
    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    n_training_steps : Number of training steps for learning the GP hypers, int

    Example usage:
        RF1 = Random_Forest_1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        RF1.optimize()  # Run optimization
        X, fX = RF1.X, RF1.fX  # Evaluated points
    """


    def __init__(
        self,
        f,
        lb ,
        ub ,
        configuration_space:dict,
        initial_design,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        random_seed = int(1e6),
        acq_funct = 'EI',
        model = 'RF',
        maximizer  = 'Sobol',
        n_folds = 5,
    ):

        # Very basic input checks
        """assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub) 
        assert np.all(ub > lb)"""
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) 
        assert isinstance(batch_size, int)
        assert max_evals > n_init and max_evals > batch_size

        # Save function information
        #Objective function
        self.f = f

        #Set a seed.
        self.seed = random_seed
        self.rng = np.random.RandomState(self.seed)

        self.config_space =  configuration_space
        # Settings
        self.n_init = n_init #Initial configurations per group
        self.max_evals = max_evals #Maxmimum evaluations in total.

        # Best configuration and the score of the best configuration.
        self.inc_score = np.inf
        self.inc_config = None

        # Save the full history
        self.X = []
        self.fX = np.array([])

        self.surrogate_time = np.array([])
        self.acquisition_time = np.array([])
        self.objective_time = np.array([])
        self.total_time = np.array([])

        #Number of current evaluations!
        self.n_evals = 0 
        
        self.n_folds = n_folds
        initial_evaluations = self.n_init * len(configuration_space)

        print('How many subspaces we got?', len(configuration_space))
        print('Initial configurations run in total',initial_evaluations)


        self.max_evals_per_fold = [25,75,150,250,500]
        print('Max Evaluations per fold',self.max_evals_per_fold)


        #Construct the Bayesian Optimization Objects per case.
        self.object_per_group = {}
        self.max_acquisitions_configs = {}
        self.max_acquisitions_score = {}


        #Initialize the X and fX dictionaries.
        for classifier_name in configuration_space:
            classifier_specific_config_space = configuration_space[classifier_name]
            self.object_per_group[classifier_name] = EI_Batch_per_group(f= self.f,lb=None,ub=None,\
                                                                                    configuration_space=classifier_specific_config_space,\
                                                                                    initial_design=initial_design,n_init=n_init,max_evals=max_evals,
                                                                                    batch_size=1,random_seed=random_seed,\
                                                                                    group_name =classifier_name,n_folds = self.n_folds )
        #Store the group that was selected at each iteration.
        self.X_group = []

    def run(self): 
        for fold in range(0,self.n_folds):
            print('Currently Running fold : ', fold)
            if fold == 0:
                for classifier_name in self.object_per_group:
                    #print('Initializing Group : ', classifier_name)
                    self.object_per_group[classifier_name].run_initial_configurations(fold)
                    #Train the surrogate model
                    self.object_per_group[classifier_name].train_surrogate()                    
                
                #This allows us to pool the performance and configurations per group
                #And add the best configuration and score to our self.X , self.fX for tracking.
                #self.compute_initial_configurations_curve()
                self.n_evals = self.n_init * len(self.object_per_group)
                self.track_initial_groups()
            else:
                for classifier_name in self.object_per_group:
                    #print('Re-run previous configurations on new fold.',classifier_name)
                    #Runs the previous configurations on the current fold
                    self.object_per_group[classifier_name].run_old_configs_on_current_fold(fold)
                    #Computes the average performance on the folds up to the current fold.
                    self.object_per_group[classifier_name].compute_avg_performance(fold)
                    #Compute the best local configuration for each group
                    self.object_per_group[classifier_name].compute_next_fold_current_inc_after_avg()
                    #Train the surrogate model.
                    self.object_per_group[classifier_name].train_surrogate()

            #At this step changed is always 1. As we find the new incumberment on the new fold.
            changed = self.compute_best_config_on_new_fold()            

            # Get the batch size.
            batch_size = 25
            total_iter = int (self.max_evals_per_fold[fold]/batch_size)
            for i in range(total_iter):

                if i != 0 :
                    for classifier_name in self.object_per_group:
                        self.object_per_group[classifier_name].train_surrogate()


                acquisition_selector_list = []
                for classifier_name in self.object_per_group:
                    tuple_of_acq_vals =self.object_per_group[classifier_name].suggest_population(self.inc_score)
                    acquisition_selector_list  += tuple_of_acq_vals

                # Sort the tuples based on the score element in descending order
                sorted_tuples = sorted(acquisition_selector_list, key=lambda x: x[1], reverse=True)


                # Retrieve the top 10 tuples
                top_10_tuples = sorted_tuples[:batch_size]
                print('Aquistion values',len(top_10_tuples),batch_size)

                #Run each group. 
                for i in top_10_tuples:
                    config  = i[0]
                    acq_val = i[1]
                    group =i[2]
                    fX_next = self.object_per_group[group].run_objective_on_previous_folds(config,fold)
                    self.object_per_group[group].compute_current_inc_after_avg()
                    self.X_group.append(group)
                
                
                #Check if new configuration is the incumberment. 
                changed = self.compute_incumberment_overall()


                        
        self.acquisition_time = np.array([0 for i in range(self.max_evals)])
        self.surrogate_time = np.array([0 for i in range(self.max_evals)])
        self.objective_time = np.array([0 for i in range(self.max_evals)])
        #Makes the final fX score progression
        self.make_X_Y()
        return self.inc_score        
    
    def make_X_Y(self):

        # Current configuration iterator in each group.
        counter_per_group = {}
        for classifier_name in self.object_per_group:
            counter_per_group[classifier_name] = 0

        #Get each group
        for group in self.X_group:
            counter = counter_per_group[group]
            self.fX = np.append(self.fX, self.object_per_group[group].fX[counter])
            counter_per_group[group]+=1

    # Compute the best configuration.
    def compute_best_config_on_new_fold(self):
        inc_score_list = []
        for classifier_name in self.object_per_group:
            inc_score_list.append((self.object_per_group[classifier_name].inc_config , self.object_per_group[classifier_name].inc_score))
        
        # Sort the list by the first element of each tuple
        # Reverse = False means that the min is first element ( LOWEST ERROR  )
        sorted_list = sorted(inc_score_list, key=lambda x: x[1],reverse=False)
        self.inc_config = sorted_list[0][0]
        self.inc_score = sorted_list[0][1]

        return 1
    
    def track_initial_groups(self):
        for i in range(0,self.n_init):
            for group_name in self.object_per_group:
                self.X_group.append(group_name)
        

    # for each classifier, get the history, 
    # and keep only the best fX for each step. (To make the error curve + the config.)
    def compute_initial_configurations_curve(self):
        df = pd.DataFrame()
        #stack fX values per group
        for classifier_name in self.object_per_group:
            df[classifier_name] = self.object_per_group[classifier_name].fX

        
        
        #print('Performance of all Groups:  ')
        #print(df)
        #Compute the incumberment per step per classifer
        self.fX= np.array(df.min(axis=1))
        #print('Minimum performance per group in line' , self.fX)
        
        #Find the group with the minimum per iteration.
        min_columns = df.idxmin(axis=1)

        min_group_per_iter = list(min_columns)
        
        #Save configurations per group.
        for i in range(len(min_group_per_iter)):
            group_name = min_group_per_iter[i]
            self.X.append(self.object_per_group[group_name].X[i])
            
    #Compute the best configuration overall.
    def compute_incumberment_overall(self):
        inc_score_list = []
        for classifier_name in self.object_per_group:
            inc_score_list.append((self.object_per_group[classifier_name].inc_config , self.object_per_group[classifier_name].inc_score))
        
        # Sort the list by the first element of each tuple
        # Reverse = False means that the min is first element ( LOWEST ERROR  )
        sorted_list = sorted(inc_score_list, key=lambda x: x[1],reverse=False)
        potential_config = sorted_list[0][0]
        potential_score = sorted_list[0][1]


        if self.inc_score > potential_score:
            self.inc_config = potential_config
            self.inc_score = potential_score
            print(f'Best score so far : {self.inc_score}')
            return 1
        return 0
