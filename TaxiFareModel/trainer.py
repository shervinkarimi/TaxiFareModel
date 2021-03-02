# imports
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LinearRegression

from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split

import pandas as pd

from memoized_property import memoized_property

import mlflow
from  mlflow.tracking import MlflowClient

MLFLOW_URI = "http://localhost:5000"
myname = "shervin"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.model = 'linear'
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipline = make_pipeline(DistanceTransformer(),StandardScaler())
        timpe_pipe = make_pipeline(TimeFeaturesEncoder('pickup_datetime'),OneHotEncoder(handle_unknown='ignore'))
        preproc_pipe = make_column_transformer(
            (dist_pipline,['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']),
            (timpe_pipe,['pickup_datetime','pickup_latitude'])
        )
        self.pipeline = make_pipeline(preproc_pipe,LinearRegression())
        

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X,self.y)
        

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        self.rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_param('model', self.model)
        self.mlflow_log_metric('rmse', self.rmse)
        
    
    #---------------------------------------------------------------------


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)
    
    


if __name__ == "__main__":

    # get data
    df = get_data(nrows=1000)
    # clean data
    df = clean_data(df, test=False)
    # set X and y
    X = df.drop(columns=['fare_amount','key'])
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)
    trainer = Trainer(X_train,y_train)

    # train
    trainer.run()
    trainer.evaluate(X_test, y_test)

    # evaluate
    print(trainer.rmse)
