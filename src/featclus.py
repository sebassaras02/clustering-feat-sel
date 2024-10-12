from joblib import Parallel, delayed

import numpy as np
import pandas as pd

from typing import List
from copy import deepcopy

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


class FeatureSelection:
    """
    This library perforns feature selection for clustering problems.
    The main idea is to shift the data and calculate the silhouette score for each feature.
    The components used are:
        - MinMaxScaler
        - PCA
        - DBSCAN
    The performance of the model is calculated by the silhouette score.

    Args:
        data: pd.DataFrame
            The data to be used in the model.
        shifts: List
            The shifts to be used in the data.

    Returns:
        pd.DataFrame: A DataFrame with the importance of each feature sorted.
    """

    def __init__(self, data: pd.DataFrame, shifts: List = [5, 10, 50], n_jobs:int = -1):
        self.data = data
        self.shifts = shifts
        self.model = Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("pca", PCA(0.8)),
                ("clustering", DBSCAN()),
            ]
        )
        self.columns = data.columns
        self.n_jobs = n_jobs

    def _shit_data(self, df: pd.DataFrame, target_column: str) -> List[pd.DataFrame]:
        """
        This function creates different dataframes based on different shifts.
        """
        data_shifted = []
        for value in self.shifts:
            df1 = deepcopy(df)
            df1[target_column] = df1[target_column].shift(value)
            df1 = df1.dropna()
            data_shifted.append(df1)
        return data_shifted

    def _train_model(self) -> dict:
        """
        This function trains the models needed in a parallelized way.
        """
        if self.n_jobs == 1:
            scores = {col: self._evaluate_feature(col) for col in self.columns}
        else:
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_feature)(col) for col in self.columns
            )
            scores = dict(zip(self.columns, scores))
        return scores
    
    def _evaluate_feature(self, col: str) -> float:
        """
        This function applies the shifts in a parallelized way.
        """
        df_to_test = self._shit_data(df=self.data, target_column=col)
        
        if self.n_jobs == 1:
            values = [self._get_score(df) for df in df_to_test]
        else:
            values = Parallel(n_jobs=self.n_jobs)(
                delayed(self._get_score)(df) for df in df_to_test
            )
        return np.mean(values)
    
    def _get_score(self, df: pd.DataFrame) -> float:
        """
        This function calculates the score of the clustering for each variable
        """
        labels = self.model.fit_predict(df)
        print(len(labels))
        print(len())
        score = silhouette_score(X=df, labels=labels)
        return score

    def get_metrics(self) -> pd.DataFrame:
        """
        This function saves the results of the metrics and sorts the results.
        """
        scores_shifted = self._train_model()
        original_score = self._get_score(df=self.data)
        final_values = {}
        for key, value in scores_shifted.items():
            final_values[key] = np.abs(original_score - value)
        df = pd.DataFrame(
            final_values.values(), columns=["Importance"], index=final_values.keys()
        ).sort_values("Importance", ascending=False)
        return df