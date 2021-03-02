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

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipline = make_pipeline(DistanceTransformer(),StandardScaler())
        timpe_pipe = make_pipeline(TimeFeaturesEncoder('pickup_datetime'),OneHotEncoder(handle_unknown='ignore'))
        preproc_pipe = make_column_transformer(
            (dist_pipline,['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude']),
            (timpe_pipe,['pickup_datetime','pickup_latitude'])
        )
        self.pipeline = make_pipeline(preproc_pipe,LinearRegression())
        return self.pipeline

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X,self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


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
    trainer.set_pipeline()
    trainer.run()

    # evaluate
    print(trainer.evaluate(X_test, y_test))
