import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode

# Read data from CSV file
data = pd.read_csv('dataset/buildings.csv')

# Configuration
k = 4 # Number of clusters

# Perform Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
data['Cluster'] = agg_clust.fit_predict(data[['esmaneKasutus', 'pindala', 'mahtBruto', 'maxKorrusteArv', 'tubadeArv', 'ehitisAlunePind', 'suletudNetoPind']])

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