import numpy as np
import pandas as pd
from typing import List
from copy import deepcopy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class FeatureSelection:
    def __init__(self, data: pd.DataFrame, shifts: List = [5, 10, 50]):
        self.data = data
        self.shifts = shifts
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("kmeans", DBSCAN())
            ]
        )
        self.columns = data.columns
        
    def _shit_data(self, df: pd.DataFrame, target_column: str) -> List[pd.DataFrame]:
        data_shifted = []
        for value in self.shifts:
            df1 = deepcopy(df)
            df1[target_column] = df1[target_column].shift(value)
            df1 = df1.dropna()
            data_shifted.append(df1)
        return data_shifted
    
    def _train_model(self):
        scores = {}
        for col in self.columns:
            df_to_test = self._shit_data(df=self.data, target_column=col)
            values = []
            for df in df_to_test:
                values.append(self._get_score(df=df))
            scores[col] = np.mean(values)
        return scores
                    
    def _get_score(self, df):
        self.model.fit(df)
        return self.model.inertia_
    
    def get_metrics(self):
        scores_shifted = self._train_model()
        original_score = self._get_score(df=self.data)
        final_values = {}
        for key, value in scores_shifted.items():
            final_values[key] = np.abs(original_score-value)
        df = pd.DataFrame(final_values.values(), columns=["Importance"], index=final_values.keys())
        return df
            
    
