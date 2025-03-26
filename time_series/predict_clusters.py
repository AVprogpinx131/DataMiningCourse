import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----- 1. Read & Preprocess the CSV -----
# Skip first 4 rows (anonymization info) and use ';' as delimiter
df = pd.read_csv('your_time_series.csv', sep=';', skiprows=4, header=0)
df.columns = ['Periood', 'consumption']
df['Periood'] = pd.to_datetime(df['Periood'], dayfirst=True)
df['consumption'] = df['consumption'].str.replace(',', '.').astype(float)

# Create date and hour columns
df['date'] = df['Periood'].dt.date
df['hour'] = df['Periood'].dt.hour

# ----- 2. Pivot Data: one row per day with 24 hourly values -----
pt = df.pivot_table(index='date', columns='hour', values='consumption', aggfunc='mean').sort_index()
pt = pt.dropna()  # Keep only complete days
dates = list(pt.index)
print("Total complete days available:", len(dates))

# ----- 3. Define helper functions -----
def compute_rmse(y_pred, y_true):
    """Calculate Root Mean Squared Error between predicted and actual values"""
    # Handle potential length mismatches
    min_len = min(len(y_pred), len(y_true))
    y_pred, y_true = y_pred[:min_len], y_true[:min_len]
    return np.sqrt(np.mean((y_pred - y_true)**2))

def compute_similarity(day1, day2):
    """Calculate the Euclidean distance between two days' hourly patterns"""
    # Handle potential length mismatches
    min_len = min(len(day1), len(day2))
    return np.sqrt(np.sum((day1[:min_len] - day2[:min_len]) ** 2))

def find_similar_days(reference_day, historical_data, n=3):
    """
    Find the n most similar days to the reference day in the historical data.
    
    Args:
        reference_day: The day pattern to compare against
        historical_data: DataFrame with historical daily patterns
        n: Number of similar days to return
    
    Returns:
        List of dates of the most similar days
    """
    similarities = []
    
    for date in historical_data.index:
        day_pattern = historical_data.loc[date].values
        similarity = compute_similarity(reference_day, day_pattern)
        similarities.append((date, similarity))
    
    # Sort by similarity (lowest distance = most similar) and get top n
    similarities.sort(key=lambda x: x[1])
    return [date for date, _ in similarities[:n]]

# ----- 4. Clustering days into 3 clusters -----
def cluster_days(n_clusters=3):
    """
    Cluster all days in the dataset into n_clusters based on hourly consumption patterns.
    
    Args:
        n_clusters: Number of clusters to create
        
    Returns:
        DataFrame with cluster labels, cluster centroids, and representative days
    """
    print(f"\n----- Clustering Days into {n_clusters} Clusters -----")
    
    # Prepare data for clustering
    X = pt.values
    
    # Standardize the data to ensure equal weighting of hours
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Get cluster centroids and convert back to original scale
    centroids_scaled = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    
    # Add cluster labels to the original data
    pt_with_clusters = pt.copy()
    pt_with_clusters['cluster'] = cluster_labels
    
    # Find representative days (closest to centroids) for each cluster
    representative_days = []
    
    for cluster_id in range(n_clusters):
        # Get days in this cluster
        cluster_days = pt_with_clusters[pt_with_clusters['cluster'] == cluster_id].drop('cluster', axis=1)
        
        # Find day closest to centroid
        min_distance = float('inf')
        closest_day = None
        
        for day in cluster_days.index:
            day_pattern = cluster_days.loc[day].values
            distance = np.sqrt(np.sum((day_pattern - centroids[cluster_id])**2))
            
            if distance < min_distance:
                min_distance = distance
                closest_day = day
        
        representative_days.append(closest_day)
    
    # Count days in each cluster
    cluster_counts = pt_with_clusters['cluster'].value_counts().sort_index()
    for cluster_id, count in enumerate(cluster_counts):
        print(f"Cluster {cluster_id}: {count} days")
    
    return pt_with_clusters, centroids, representative_days

def plot_clusters(pt_with_clusters, centroids, representative_days, mode='centroids'):
    """
    Plot cluster patterns.
    
    Args:
        pt_with_clusters: DataFrame with cluster assignments
        centroids: Cluster centroids
        representative_days: Representative days for each cluster
        mode: 'centroids' to plot centroids or 'representatives' to plot representative days
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['b', 'r', 'g']
    hours = range(24)
    
    if mode == 'centroids':
        # Plot centroids
        for i, centroid in enumerate(centroids):
            plt.plot(hours[:len(centroid)], centroid[:len(hours)], 
                     f'{colors[i]}-', linewidth=2, 
                     label=f'Cluster {i} (n={pt_with_clusters["cluster"].value_counts().get(i, 0)})')
        
        title = 'Cluster Centroids: Average Consumption Patterns'
    else:
        # Plot representative days
        for i, day in enumerate(representative_days):
            day_values = pt.loc[day].values
            plt.plot(hours[:len(day_values)], day_values[:len(hours)], 
                     f'{colors[i]}-', linewidth=2, 
                     label=f'Cluster {i} Rep: {day}')
        
        title = 'Representative Days for Each Cluster'
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Energy Consumption (kWh)')
    plt.title(title)
    plt.xticks(range(0, 24, 2))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    filename = 'cluster_centroids.png' if mode == 'centroids' else 'cluster_representatives.png'
    plt.savefig(filename)
    plt.show()
    
    print(f"Plot saved as '{filename}'")

# ----- 5. Main execution -----
def main():
    # Cluster days and plot results
    pt_with_clusters, centroids, representative_days = cluster_days(n_clusters=3)
    
    # Plot cluster centroids
    plot_clusters(pt_with_clusters, centroids, representative_days, mode='centroids')
    
    # Plot representative days
    plot_clusters(pt_with_clusters, centroids, representative_days, mode='representatives')

if __name__ == "__main__":
    main()