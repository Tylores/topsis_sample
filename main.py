import numpy as np 
import pandas as pd 
from topsis import TOPSIS

def rank (data: pd.DataFrame, similarity: np.ndarray) -> pd.DataFrame:
    print(similarity)
    data['similarity'] = similarity
    return data.sort_values(by='similarity', ascending=False)

if __name__ == "__main__":
    data = pd.read_csv('data/data.csv', index_col='device', na_values= ' ')
    
    # Step 1: Normalizing
    topsis = TOPSIS(data)
    
    # Step 2: Apply Weights
    weights = [0.2, 0.3, 0.5]
    weighted_norm = topsis.weighted_norm([0.2, 0.3, 0.5])
    
    # Step 3: Identify PIS/NIS
    costs = [0,1,2]
    pis, nis = topsis.identify_ideals(weighted_norm, costs)
    
    # Step 4: Calculate Seperation measures
    dist_pis, dist_nis = topsis.calculate_seperation(weighted_norm, pis, nis)
    
    # Step 5: Calculate Similarity
    similarity = topsis.calculate_similarity(dist_pis, dist_nis)
    
    # Step 6: Rankings
    rankings = rank(data, similarity)
    print(rankings)
    