import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import mode

# Read data from CSV file
data = pd.read_csv('dataset/buildings.csv')  # Ensure this path matches the location of your CSV file

# Configuration
k = 4 # Number of clusters

# Perform K-Means clustering with explicitly setting n_init to suppress the warning
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['esmaneKasutus', 'pindala', 'mahtBruto', 'maxKorrusteArv', 'tubadeArv', 'ehitisAlunePind', 'suletudNetoPind']])

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