import numpy as np
import pandas as pd
from typing import Tuple
from scipy import sparse
from scipy.stats import rankdata # for ranking the candidates

class TOPSIS():
    """Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)
    """

    def __init__(self, data: pd.DataFrame):
        """Store data for TOPSIS

        Args:
            data (pd.DataFrame): indices are candidates and columns are desired attributes
        """
        self._candidates = data.index.array.to_numpy()
        self._attributes = data.columns.to_numpy()
        self._data = data.to_numpy()
        self._m = len(self._data)
        self._n = len(self._attributes)
        
        self._data = self.__normalize()
        
    def __mask_missing(self):
        """Mask all missing values to prevent issues with matrix operations
        """
        self._data = np.ma.array(self._data, mask=np.isnan(self._data))
        
    def __normalize(self) -> np.ndarray:
        """Normalize data

        Returns:
            np.ndarray: normalized data
        """
        self.__mask_missing()
        norm = np.sqrt(np.nansum(np.square(self._data)))
        return self._data/norm
        
    def weighted_norm(self, weights: np.ndarray) -> np.ndarray:
        """Apply unit weights to normalized data

        Args:
            weights (np.ndarray): unit attributed weights
            
        Returns:
            np.ndarray: weighted data
        """
        return self.__normalize()*weights
        
    def identify_ideals(self, data: np.ndarray, costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Identify Positive/Negative Ideal Solutions (PIS) and (NIS)

        Args:
            data (np.ndarray): _description_
            costs (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        pis = np.zeros(self._n)
        nis = np.zeros(self._n)
        for j in range(self._n):
            attribute = data[:,j]
            max_val = np.max(attribute)
            min_val = np.min(attribute)
            
            # minimize costs
            if j in costs:
                pis[j] = min_val
                nis[j] = max_val
            else:
                pis[j] = max_val
                nis[j] = min_val
                
        return [pis, nis]
    
    def calculate_seperation(self, data: np.ndarray, pis: np.ndarray, nis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate euclidean distance between data and the PIS/NIS

        Args:
            data (np.ndarray): _description_
            pis (np.ndarray): _description_
            nis (np.ndarray): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """
        dist_pis = np.zeros(self._m)
        dist_nis = np.zeros(self._m)
        
        for i in range(self._m):
            dist_pis[i] = np.linalg.norm(data[i] - pis)
            dist_nis[i] = np.linalg.norm(data[i] - nis)
            
        return [dist_pis, dist_nis]
    
    def calculate_similarity(self, dist_pis: np.ndarray, dist_nis: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            data (np.ndarray): _description_
            dist_pis (np.ndarray): _description_
            dist_nis (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        similarity = np.zeros(self._m)
        
        for i in range(self._m):
            similarity[i] = dist_nis[i] / (dist_pis[i] + dist_nis[i])
            
        return similarity