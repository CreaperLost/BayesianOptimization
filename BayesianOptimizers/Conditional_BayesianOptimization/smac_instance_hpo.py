"""
                per_instance_scenario = Scenario(
                        configspace,
                        name='Dataset'+str(task_id),
                        output_directory = 'per_instance_smac/'+repo,
                        n_trials=max_evals,  # We want to try max 5000 different trials
                        min_budget=1,  # Use min 1 fold.
                        max_budget=10,  # Use max 10 folds. 
                        instances=[0,1,2,3,4,5,6,7,8,9],
                        deterministic=True,seed=seed,n_workers = 4
                    )
                # Create our SMAC object and pass the scenario and the train method
                per_instance_smac = MFFacade(
                    per_instance_scenario,
                    objective_function_per_fold,
                    overwrite=True,
                )

                # Now we start the optimization process
                instance_incumbent = per_instance_smac.optimize()
                print(instance_incumbent)
                # Let's calculate the cost of the incumbent
                instance_incumbent_cost = per_instance_smac.validate(instance_incumbent)
                print(f"Instance Incumbent cost: {instance_incumbent_cost}")
                #The file path for current optimizer.
                """