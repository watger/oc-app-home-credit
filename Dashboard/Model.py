# 1. Library imports
import pandas as pd 
#from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import re


# 2. Class which describes a single flower measurements
class HomeCreditRisk(BaseModel):
    id_client : int



# 3. Class for training the model and making predictions
class HCRModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        self.df = pd.read_csv('../Data/data_200.csv',
                              #low_memory=False, 
                              #error_bad_lines = False, 
                              #engine ='python'
                             )
        self.df = self.df.head(100)
        self.df = self.df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        #self.df = joblib.load('https://github.com/watger/OC_Projects.git/Projet 7/data.csv')
        self.model_fname_ = '../Data/model_f.joblib'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            print("model_f not found")
            #self.model = self._train_model()
            #joblib.dump(self.model, self.model_fname_)
        
        self.scaler_fname_ = '../Data/scaler.joblib'
        try:
            self.scaler = joblib.load(self.scaler_fname_)
        except Exception as _:
            print("scaler not found")   
         
            
        
        X = self.df.drop(columns=["TARGET","SK_ID_CURR"])#,"YEARS_BIRTH"
        y = self.df.TARGET
        

        self.model.fit(X,y)
        

    # 4. Perform model training using the RandomForest classifier
    def _train_model(self):
        X = self.df.drop('TARGET', axis=1)
        y = self.df['TARGET']
        lgbmc = lgb.LGBMClassifier(
            nthread=4,
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=70,
            max_depth=15,
            reg_alpha=0.3,
            reg_lambda=0.3,
            n_jobs=10,
        )
        model = lgbmc.fit(X, y)
        return model

    # 5. Make a prediction based on the user-entered data
    #    Returns the predicted species with its respective probability
    def predict_risks(self, id_client):
        id_client = float(id_client)
        #data = self.df[self.df.SK_ID_CURR == id_client]
        data = self.df
        #data = data.reset_index(drop=True)
        index = data.index[data.SK_ID_CURR == id_client][0]
        data_in = data.drop(columns=["TARGET","SK_ID_CURR"])#,"YEARS_BIRTH"
        #data_in = self.scaler.transform(data_in)
        #prediction = self.model.predict(data_in)[index]
        #probability = self.model.predict_proba(data_in)
        probability = self.model.predict_proba(data_in)[index][0]#.max()##[id_client,1]
        #print(probability)
        #print(type(id_client))
        #print(self.model.predict_proba(data_in))
        
        if probability < 0.91 :
            prediction = "Rejected"
        else :
            prediction = "Approuved"
        
        return prediction, round(probability,3)#, index, id_client#, data #prediction

    """
    # 5. Make a prediction based on the user-entered data
    #    Returns the predicted species with its respective probability
    def predict_risks(self, id_client_dict):
        id_client = list(id_client_dict.values())
        print(id_client)
        data = self.df[self.df.SK_ID_CURR == id_client[0]]
        data_in = data.drop(columns=["TARGET","SK_ID_CURR"])
        print(data_in)
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability
    """