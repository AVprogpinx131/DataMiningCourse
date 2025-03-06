import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.stats import mode

# Read data from CSV file
data = pd.read_csv('dataset/buildings.csv')  # Ensure this path matches the location of your CSV file

# Configuration
k = 4 # Number of clusters

# Perform Spectral Clustering
spectral_clust = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
data['Cluster'] = spectral_clust.fit_predict(data[['esmaneKasutus', 'pindala', 'mahtBruto', 'maxKorrusteArv', 'tubadeArv', 'ehitisAlunePind', 'suletudNetoPind']])

# Print out clusters and their descriptions
for cluster in range(k):
    cluster_data = data[data['Cluster'] == cluster]
    print(f'Cluster {cluster + 1}')
    buildings = cluster_data['Buildings'].tolist()
    print('Buildings:', buildings)
    
    # Calculate mean for numerical data
    print('Average Values:')
    for column in ['esmaneKasutus', 'pindala', 'mahtBruto', 'maxKorrusteArv', 'tubadeArv', 'ehitisAlunePind', 'suletudNetoPind']:
        print(f'{column}: {cluster_data[column].mean():.2f}')
    
    print('')  # For better readability