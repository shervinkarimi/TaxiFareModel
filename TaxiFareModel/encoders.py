from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized, extract_time_features
import pandas as pd


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """Extract the day of week (dow), the hour, the month and the year from a
    time column."""
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'"""
        return extract_time_features(X)[['dow','hour','month','year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """Compute the haversine distance between two GPS points."""
    def __init__(self, 
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude", 
                 end_lat="dropoff_latitude", 
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        """Returns a copy of the DataFrame X with only one column: 'distance'"""
        return pd.DataFrame(haversine_vectorized(X, 
             start_lat=self.start_lat,
             start_lon=self.start_lon,
             end_lat=self.end_lat,
             end_lon=self.end_lon))
