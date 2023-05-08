import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from jadbio_api.api_client import ApiClient
import os
from hpobench.util.data_manager import DataManager
from get_pass import get_pass

class JadDataManager(DataManager):

    def __init__(self, task_id: int,
                 data_path: Union[str, Path, None] = None,
                 global_seed: Union[int, None] = 1,
                 n_folds :int = 5,
                 use_holdout = False):

        self.task_id = task_id
        self.global_seed = global_seed

        self.train_X = []
        self.valid_X = []
        self.test_X = None
        self.train_y = []
        self.valid_y = []
        self.test_y = None
        self.train_idx = None
        self.test_idx = None
        self.task = None
        self.dataset = None
        self.preprocessor = None
        self.lower_bound_train_size = None
        self.n_classes = None
        self.n_folds = n_folds
        self.use_holdout = use_holdout

        if data_path is None:
            self.data_path = 'Jad_Datasets/Jad'
            #data_path = config_file.data_dir / "OpenML"
        else:
            self.data_path = data_path
        
        
        
        
        #self.data_path = Path(data_path)
        #openml.config.set_cache_directory(str(self.data_path))

        super(JadDataManager, self).__init__()

   
        
        

    # pylint: disable=arguments-differ
    """@lockutils.synchronized('not_thread_process_safe', external=True,
                            lock_path=f'{config_file.cache_dir}/openml_dm_lock', delay=0.2)"""
    def load(self, verbose=False):
        """Fetches data from JadBio and initializes the train-validation-test data splits

        The validation set is fixed till this function is called again or explicitly altered
        """
  
        if verbose:
            self.logger.debug(self.task)
            self.logger.debug(self.dataset)

        #Check if we locally have the specific dataset.
        data_set_path = self.data_path + "/datasets/" + str(self.task_id)
        successfully_loaded = self.try_to_load_data(data_set_path)

        

        if successfully_loaded:
            self.logger.info(f'Successfully loaded the preprocessed splits from '
                             f'{data_set_path}')
            return


        #Should recheck
        try:
            Path(data_set_path).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print("Folder is already there")
        # If the data is not available, download it.
        self.__download_data(file_path = data_set_path,verbose=verbose)



        

        # Save the preprocessed splits to file for later usage.
        self.generate_openml_splits(data_set_path)

        return


    def find_number_of_target(self,train_y=None,valid_y=None):
        return pd.concat((train_y,valid_y),axis=0).nunique()

    def try_to_load_data(self, data_path: str) -> bool:
        path_str = "{}_{}_{}.parquet.gzip"
        #For test.
        if self.use_holdout == True:
            path_str2 = "{}_{}.parquet.gzip"
        try:
            for fold in range(self.n_folds) :
                self.train_X.append( pd.read_parquet(data_path + path_str.format("train", "x",str(fold))).to_numpy())
                self.train_y.append( pd.read_parquet(data_path + path_str.format("train", "y",str(fold))).squeeze(axis=1))
                self.valid_X.append( pd.read_parquet(data_path + path_str.format("valid", "x",str(fold))).to_numpy())
                self.valid_y.append(pd.read_parquet(data_path + path_str.format("valid", "y",str(fold))).squeeze(axis=1))
                
            #Don't Load test-data any more --
            if self.use_holdout == True:
                self.test_X = pd.read_parquet(data_path + path_str2.format("test", "x")).to_numpy()
                self.test_y = pd.read_parquet(data_path + path_str2.format("test", "y")).squeeze(axis=1)

            self.n_classes = self.find_number_of_target(self.train_y[0],self.valid_y[0])
        except FileNotFoundError:
            return False
        return True


    def find_categorical_columns(self,X):

        cat_ind_bool  = [X.columns.get_loc(col) for col in X.columns.tolist() if len(np.unique(X[[col]])) <= 2]
        cat_ind_cat  = [X.columns.get_loc(col) for col in X.select_dtypes(include=['object']).columns.tolist() ]
        cat_indx =list(set(cat_ind_bool + cat_ind_cat))

        all_col_loc = [X.columns.get_loc(col) for col in X.columns.tolist()]

        continuous_indx = [x for x in all_col_loc if x not in cat_indx]

        return cat_indx,continuous_indx

    def bool_dtypes_to_int(self,X):
        X_new = X.copy()

        bool_idx  = [col for col in X.select_dtypes(include=['bool']).columns.tolist() ]

        X_new[bool_idx] = X_new[bool_idx].astype(int)

        return X_new

    def preprocess_data(self,dataset):
        df = dataset.copy()
        df.drop('gr.gnosisda-1',axis=1,inplace=True)
        X = df.drop(['target'],axis=1)
        y = df['target']
        #Categorical_ind is experimental
        #Filter by unique.
        #set a threshold

        
        #categorical_ind = [X.columns.get_loc(col) for col in X.columns.tolist() if len(np.unique(X[[col]])) <= threshold]
        #continuous_ind  = [X.columns.get_loc(col) for col in X.select_dtypes(exclude=['object']).columns.tolist() ]
        #continuous_ind  = [X.columns.get_loc(col) for col in X.columns.tolist() if len(np.unique(X[[col]])) > threshold]


        categorical_ind,continuous_ind = self.find_categorical_columns(X)


        X = self.bool_dtypes_to_int(X)

        return X,y,categorical_ind,continuous_ind

    def __download_data(self,file_path:str, verbose: bool):
        assert file_path!=None
        #self.logger.info('Start to download the OpenML dataset')
                
        tmp_file_loc= file_path + '/' + 'data.csv' #os.getcwd() + '/Jad_Temp/'+ 'dataset'+ str(self.task_id) + '.csv'
        if os.path.exists(tmp_file_loc):
            print('Dataset Exists on DB.')
        else:
            ip, email, password =  get_pass('Good')
            self.Client = ApiClient(ip, email, password)
            self.Client.project.download_dataset(self.task_id,tmp_file_loc)

        dataset = pd.read_csv(tmp_file_loc)
        

        X, y, categorical_ind,continuous_ind = self.preprocess_data(dataset)
        self.n_classes = y.nunique()

        #Label encode y.
        labelencoder = LabelEncoder()
        y = pd.Series(labelencoder.fit_transform(y))
        #assert Path(self.dataset.data_file).exists(), f'The datafile {self.dataset.data_file} does not exists.'

        cat_idx  = categorical_ind
        cont_idx = continuous_ind
        

        #If we want to use hold-out set or not. :)
        if self.use_holdout == True:
            # splitting dataset into train and test (10% test)
            # train-test split is fixed for a task and its associated dataset (from OpenML)
            self.train_idx, self.test_idx = self.task.get_train_test_split_indices()
            train_x = X.iloc[self.train_idx]
            train_y = y.iloc[self.train_idx]

            self.test_X = X.iloc[self.test_idx]
            self.test_y = y.iloc[self.test_idx]
        else:
            train_x = X
            train_y = y

        # splitting training into training and validation
        # validation set is fixed as per the global seed independent of the benchmark seed

        #Instead of this please do cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds,shuffle=True,random_state=self.global_seed)
        for train_index, val_index in skf.split(train_x, train_y):
            X_train = train_x.iloc[train_index]
            y_train = train_y.iloc[train_index]
            X_val  = train_x.iloc[val_index]
            y_val  = train_y.iloc[val_index]
            self.train_X.append(X_train)
            self.train_y.append(y_train)
            self.valid_X.append(X_val)
            self.valid_y.append(y_val)



        # preprocessor to handle missing values, categorical columns encodings,
        # and scaling numeric columns

        #,
        self.preprocessor = make_pipeline(
            ColumnTransformer([
                (
                    "cat",
                    make_pipeline(SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(sparse=False, handle_unknown="ignore")),
                    cat_idx,
                ),
                (
                    "cont",
                    make_pipeline(SimpleImputer(strategy="median"),
                                  StandardScaler()),
                    cont_idx,
                )
            ])
        )
        if verbose:
            self.logger.debug("Shape of data pre-preprocessing: {}".format(self.train_X[0].shape))



        #Get back the training dataset, by combining the  training-validation.
        #Learn the preprocess and apply it to the test set. (FIT THE PROCEDURE TO WHOLE DATA)
        
        #Keep a training set before transformations
        if self.use_holdout == True:
            train_X = np.vstack((self.train_X[0], self.valid_X[0]))
            self.preprocessor.fit(train_X)

            #Do appropriate transformation for the test set.
            self.test_X = self.preprocessor.transform(self.test_X)
            self.test_y = self._convert_labels(self.test_y)
        

        for fold in range(self.n_folds):
            # preprocessor fit only on the training set
            self.train_X[fold] = self.preprocessor.fit_transform(self.train_X[fold])
            # applying preprocessor built on the training set, across validation and test splits
            self.valid_X[fold] = self.preprocessor.transform(self.valid_X[fold])
            # converting boolean labels to strings
            self.train_y[fold] = self._convert_labels(self.train_y[fold])
            self.valid_y[fold] = self._convert_labels(self.valid_y[fold])


    
        """# Similar to (https://arxiv.org/pdf/1605.07079.pdf)
        # use 10 times the number of classes as lower bound for the dataset fraction
        self.lower_bound_train_size = (10 * self.n_classes) / self.train_X.shape[0]
        self.lower_bound_train_size = np.max((1 / 512, self.lower_bound_train_size))"""

        if verbose:
            self.logger.debug("Shape of data post-preprocessing: {}".format(self.train_X[0].shape), "\n")
            self.logger.debug("\nTraining data (X, y): ({}, {})".format(self.train_X[0].shape, self.train_y[0].shape))
            self.logger.debug("Validation data (X, y): ({}, {})".format(self.valid_X[0].shape, self.valid_y[0].shape))
            self.logger.debug("Test data (X, y): ({}, {})".format(self.test_X.shape, self.test_y.shape))
            self.logger.debug("\nData loading complete!\n")

    def generate_openml_splits(self, data_path):
        """ Store the created splits to file for later use… """
        self.logger.info(f'Save the splits to {data_path}')

        path_str = "{}_{}_{}.parquet.gzip"

        #For test.
        if self.use_holdout == True:
            path_str2 = "{}_{}.parquet.gzip"

        
        label_name = 'target'
        for fold in range(self.n_folds):
            colnames = np.arange(self.train_X[fold].shape[1]).astype(str)
            pd.DataFrame(self.train_X[fold], columns=colnames).to_parquet(data_path + path_str.format("train", "x",str(fold)))
            self.train_y[fold].to_frame(label_name).to_parquet(data_path + path_str.format("train", "y",str(fold)))
            pd.DataFrame(self.valid_X[fold], columns=colnames).to_parquet(data_path + path_str.format("valid", "x",str(fold)))
            self.valid_y[fold].to_frame(label_name).to_parquet(data_path + path_str.format("valid", "y",str(fold)))
        
        # 1 Hold-out Test set.
        if self.use_holdout == True:
            colnames = np.arange(self.test_X.shape[1]).astype(str)
            pd.DataFrame(self.test_X, columns=colnames).to_parquet(data_path + path_str2.format("test", "x"))
            self.test_y.to_frame(label_name).to_parquet(data_path + path_str2.format("test", "y"))

    @staticmethod
    def _convert_labels(labels):
        """Converts boolean labels (if exists) to strings
        """
        label_types = list(map(lambda x: isinstance(x, bool), labels))
        if np.all(label_types):
            _labels = list(map(lambda x: str(x), labels))
            if isinstance(labels, pd.Series):
                labels = pd.Series(_labels, index=labels.index)
            elif isinstance(labels, np.array):
                labels = np.array(labels)
        return labels
 