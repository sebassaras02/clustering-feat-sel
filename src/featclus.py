import numpy as np 
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.pipeline import Pipeline

class FeatureSelectionClustering:

    def __init__(self, data):
        self.data =  data

    def _define_pipeline(self):
        pipeline = Pipeline(
            steps=[
                ('scaler', MinMaxScaler()),
                ('clustering', DBSCAN())
            ]
        )
