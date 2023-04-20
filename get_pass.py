import pandas as pd 

import os 



def get_pass(relative = None):
    if relative == None:
        pass_df = pd.read_csv(os.path.abspath('..')+ '/pass.csv')
    else:
        pass_df = pd.read_csv('pass.csv')
    return pass_df['IP'].values[0], pass_df['email'].values[0], pass_df['pass'].values[0]

