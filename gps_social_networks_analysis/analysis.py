import os
import gpxpy
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime, timedelta
import zipfile
import io
import gc
from matplotlib.lines import Line2D

# Function to parse GPX files from a zip archive
def parse_gpx_files_from_zip(zip_path):
    all_points = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        gpx_files = [f for f in zip_ref.namelist() if f.endswith('.gpx')]
        
        print(f"Found {len(gpx_files)} GPX files in archive")
        for gpx_file in gpx_files:
            # Read and parse the GPX file
            with zip_ref.open(gpx_file) as f:
                gpx_content = f.read()
                gpx = gpxpy.parse(io.BytesIO(gpx_content))

                # Try multiple date extraction methods
                date = extract_date_from_gpx(gpx, gpx_file)

                day_points = []
                points_with_time = 0
                for track in gpx.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            point_time = None
                            # Try getting time from the point itself
                            if hasattr(point, 'time') and point.time:
                                point_time = point.time
                                points_with_time += 1
                                # Use point's date if available and we don't have a good date yet
                                if date == "unknown" and point_time:
                                    date = point_time.strftime('%Y-%m-%d')

                            day_points.append({
                                'date': point_time.strftime('%Y-%m-%d') if point_time else date,
                                'timestamp': point_time,
                                'latitude': point.latitude,
                                'longitude': point.longitude
                            })
                
                print(f"Processed {gpx_file}: {len(day_points)} points, {points_with_time} with timestamps")
                
                # Process one day at a time to save memory
                if day_points:
                    day_df = pd.DataFrame(day_points)
                    day_stays = process_day_points(day_df, date)
                    if not day_stays.empty:
                        all_points.extend(day_stays.to_dict('records'))
                        print(f"  Found {len(day_stays)} stay locations")
                    else:
                        print(f"  No stay locations identified")

                    # Clean up to free memory
                    del day_points
                    del day_df
                    gc.collect()

    if all_points:
        return pd.DataFrame(all_points)
    return pd.DataFrame()

# Helper function to extract date from GPX file
def extract_date_from_gpx(gpx, gpx_file):
    """Extract date from GPX file using multiple methods"""
    # Method 1: Try GPX metadata time
    if gpx.time:
        return gpx.time.strftime('%Y-%m-%d')
    
    # Method 2: Try first point's time
    for track in gpx.tracks:
        for segment in track.segments:
            if segment.points and hasattr(segment.points[0], 'time') and segment.points[0].time:
                return segment.points[0].time.strftime('%Y-%m-%d')
    
    # Method 3: Try filename pattern
    try:
        filename = os.path.basename(gpx_file)
        # Check different filename patterns
        import re
        date_match = re.search(r'(\d{8})', filename)
        if date_match:
            date_str = date_match.group(1)
            return datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error extracting date from filename: {e}")
    
    # Use current date as last resort
    print(f"Warning: Using today's date for {gpx_file} as no valid date found")
    return datetime.now().strftime('%Y-%m-%d')

# Process a day's worth of points to find stay locations
def process_day_points(df, date, eps=0.0005, min_samples=3, min_stay_minutes=2):
    """
    Find locations where the user stayed for some time using DBSCAN clustering
    
    Args:
        df: DataFrame with GPS points
        date: Date string for these points
        eps: Maximum distance between points in the same cluster
        min_samples: Minimum points to form a cluster
        min_stay_minutes: Minimum duration to consider as a stay
    
    Returns:
        DataFrame with identified stay locations
    """
    if df.empty:
        return pd.DataFrame()
    
    print(f"Processing {len(df)} points for {date}")
    
    # Ensure timestamp is properly parsed
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # For very large datasets, downsample to improve performance
    if len(df) > 1000:
        print(f"  Downsampling from {len(df)} to {len(df)//10} points")
        df = df.iloc[::10]  # Keep every 10th point
    
    # Apply DBSCAN clustering to find locations with multiple points
    coords = df[['latitude', 'longitude']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['cluster'] = clustering.labels_
    
    # Find stay locations (clusters where you remained for some time)
    stays = []
    
    # Process each cluster
    for cluster_id in sorted(df[df['cluster'] != -1]['cluster'].unique()):
        cluster_points = df[df['cluster'] == cluster_id].sort_values('timestamp')
        
        if len(cluster_points) >= min_samples:
            # Calculate stay duration if timestamps are available
            if 'timestamp' in cluster_points and not cluster_points['timestamp'].isna().all():
                first_time = cluster_points['timestamp'].iloc[0]
                last_time = cluster_points['timestamp'].iloc[-1]
                
                if pd.notna(first_time) and pd.notna(last_time):
                    duration_minutes = (last_time - first_time).total_seconds() / 60
                    
                    # Only consider as a stay if duration exceeds minimum
                    if duration_minutes >= min_stay_minutes:
                        avg_lat = cluster_points['latitude'].mean()
                        avg_lon = cluster_points['longitude'].mean()
                        
                        stays.append({
                            'date': date,
                            'latitude': avg_lat,
                            'longitude': avg_lon,
                            'duration_minutes': duration_minutes,
                            'arrival': first_time,
                            'departure': last_time,
                            'point_count': len(cluster_points)
                        })
    
    return pd.DataFrame(stays)

# Group nearby stay locations into places
def assign_place_ids(stays_df, eps=0.0005, min_samples=1):
    """
    Group nearby stay locations into places using DBSCAN clustering
    
    Args:
        stays_df: DataFrame with latitude and longitude of stays
        eps: Maximum distance between points in the same cluster
        min_samples: Minimum points to form a cluster
    
    Returns:
        DataFrame with place_id column added
    """
    if stays_df.empty:
        return stays_df
    
    # Make a copy to avoid modifying the original
    df = stays_df.copy()
    
    # Extract coordinates for clustering
    coords = df[['latitude', 'longitude']].values
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
    # Add place ID (cluster label)
    df['place_id'] = clustering.labels_
    
    # Count points in each cluster
    cluster_counts = df['place_id'].value_counts()
    print(f"Found {len(cluster_counts)} distinct places")
    print(f"Largest cluster has {cluster_counts.max()} visits")
    print(f"Noise points (no cluster): {sum(df['place_id'] == -1)}")
    
    # Change noise points (-1) to unique IDs starting from max cluster ID + 1
    next_id = df['place_id'].max() + 1
    for idx in df[df['place_id'] == -1].index:
        df.at[idx, 'place_id'] = next_id
        next_id += 1
    
    return df

# Create a transition graph between places
def create_transition_graph(stays_df):
    """
    Create a graph of transitions between places
    
    Args:
        stays_df: DataFrame with place_id and arrival timestamp
        
    Returns:
        Dictionary mapping (from_place, to_place) to transition count
    """
    if stays_df.empty:
        return {}

    transitions = []
    
    # Make a copy to avoid modifying the original
    df = stays_df.copy()
    
    # Sort by arrival time if available, otherwise by index
    if 'arrival' in df.columns:
        df = df.sort_values('arrival')
    
    # Group by date
    for date, group in df.groupby('date'):
        places = group['place_id'].tolist()
        # Record transitions within the day
        for i in range(len(places) - 1):
            from_place = places[i]
            to_place = places[i+1]
            if from_place != to_place:  # Don't count transitions to the same place
                transitions.append((from_place, to_place))

    # Count frequencies
    transition_counts = {}
    for from_place, to_place in transitions:
        key = (from_place, to_place)
        if key in transition_counts:
            transition_counts[key] += 1
        else:
            transition_counts[key] = 1
    
    # Print summary stats
    print(f"Found {len(transition_counts)} unique transitions between places")
    if transition_counts:
        print("Top 3 most frequent transitions:")
        for (src, dest), count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  Place {src} → Place {dest}: {count} times")

    return transition_counts

# Draw the transition graph between places
def draw_transition_graph(stays_df, transition_counts, top_n=20):
    """
    Create a visualization of transitions between places
    
    Args:
        stays_df: DataFrame with place_id and duration_minutes
        transition_counts: Dictionary of transition counts
        top_n: Number of top places to include in the graph
    
    Returns:
        Matplotlib figure
    """
    if stays_df.empty or not transition_counts:
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.5, "No transitions available", ha='center', va='center')
        plt.axis('off')
        return plt

    # Create directed graph
    G = nx.DiGraph()

    # Count visits to each place
    place_visits = stays_df['place_id'].value_counts()

    # Get top N most visited places
    top_places = place_visits.nlargest(min(top_n, len(place_visits))).index.tolist()

    # Add nodes for top places
    for place_id in top_places:
        visit_count = place_visits[place_id]
        G.add_node(place_id, size=visit_count)

    # Filter transitions to top places
    filtered_transitions = {(src, dest): count for (src, dest), count 
                           in transition_counts.items() 
                           if src in top_places and dest in top_places}

    # Add edges with weights based on transition counts
    for (src, dest), count in filtered_transitions.items():
        G.add_edge(src, dest, weight=count)

    # Cluster places by duration of stays for color coding
    place_durations = stays_df.groupby('place_id')['duration_minutes'].mean().loc[top_places]
    
    # Use KMeans to categorize durations
    kmeans = KMeans(n_clusters=min(3, len(place_durations)), random_state=42)
    if len(place_durations) > 0:
        place_categories = kmeans.fit_predict(place_durations.values.reshape(-1, 1))
        
        # Get the centers to determine which is short/medium/long
        centers = kmeans.cluster_centers_.flatten()
        duration_order = np.argsort(centers)
        
        # Map the categories to short, medium, long based on centers
        category_map = {duration_order[0]: 0, 
                       duration_order[min(1, len(duration_order)-1)]: 1, 
                       duration_order[min(2, len(duration_order)-1)]: 2}
        
        # Remap categories based on duration
        place_categories = np.array([category_map[cat] for cat in place_categories])
    else:
        place_categories = []

    # Map categories to colors
    category_colors = {0: 'red', 1: 'blue', 2: 'green'}  # short, medium, long
    node_colors = {place_id: category_colors[cat] for place_id, cat in zip(top_places, place_categories)}

    # Draw the graph
    plt.figure(figsize=(12, 10))

    if len(G.nodes) > 0:
        # Use spring layout for nice visualization
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        
        # Node sizes proportional to visit count
        node_sizes = [G.nodes[node]['size'] * 200 for node in G.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=[node_colors.get(node, 'gray') for node in G.nodes],
            alpha=0.8
        )
        
        # Draw edges with thickness based on transition frequency
        if len(G.edges) > 0:
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            
            # Different colors for one-way vs bidirectional edges
            curved_edges = [edge for edge in G.edges() if (edge[1], edge[0]) in G.edges()]
            straight_edges = [edge for edge in G.edges() if (edge[1], edge[0]) not in G.edges()]
            
            # Draw straight edges (one-way transitions)
            if straight_edges:
                nx.draw_networkx_edges(G, pos, edgelist=straight_edges, 
                                     width=[(G[u][v]['weight'] / max_weight) * 5 for u, v in straight_edges],
                                     edge_color='gray', alpha=0.7, arrows=True, arrowsize=15)
            
            # Draw curved edges (bidirectional transitions)
            if curved_edges:
                nx.draw_networkx_edges(G, pos, edgelist=curved_edges,
                                     width=[(G[u][v]['weight'] / max_weight) * 5 for u, v in curved_edges],
                                     edge_color='darkblue', alpha=0.7, arrows=True, arrowsize=15,
                                     connectionstyle='arc3,rad=0.2')
        
        # Add node labels
        labels = {node: f"{node}\n({G.nodes[node]['size']})" for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold', 
                              bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Add a legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Short Stays'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Medium Stays'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Long Stays'),
            Line2D([0], [0], color='gray', lw=2, label='One-way transition'),
            Line2D([0], [0], color='darkblue', lw=2, label='Two-way transition')
        ]
        plt.legend(handles=legend_elements, title="Legend", loc='upper right')
        
        plt.title(f"Transitions Between Top {len(top_places)} Most Visited Places", pad=20, fontsize=14)
    else:
        plt.text(0.5, 0.5, "No significant place transitions detected", ha='center', va='center')
    
    plt.axis('off')
    plt.tight_layout()
    return plt

# Generate a comprehensive visit summary
def generate_visit_summary(stays_df, visit_dates):
    """
    Generate a text summary of visit patterns
    
    Args:
        stays_df: DataFrame with stay information
        visit_dates: Dictionary mapping place_ids to lists of dates
        
    Returns:
        String containing formatted summary text
    """
    if stays_df.empty:
        return "No data available for analysis"
    
    # Calculate key metrics
    total_unique_places = len(visit_dates)
    total_visits = len(stays_df)
    
    # Calculate true new vs. repeated visits
    places_seen = set()
    new_visits = 0
    repeated_visits = 0
    
    # Sort by timestamp if available
    if 'arrival' in stays_df.columns:
        df_sorted = stays_df.sort_values('arrival')
    else:
        df_sorted = stays_df
    
    for _, row in df_sorted.iterrows():
        place_id = row['place_id']
        if place_id in places_seen:
            repeated_visits += 1
        else:
            new_visits += 1
            places_seen.add(place_id)
    
    # Count visits per place
    visit_counts = {}
    for place_id, dates in visit_dates.items():
        visit_counts[place_id] = len(dates)
    
    # Count places by visit frequency
    frequency_counts = {}
    for count in visit_counts.values():
        if count in frequency_counts:
            frequency_counts[count] += 1
        else:
            frequency_counts[count] = 1
    
    # Calculate places visited once vs multiple times
    visited_once = sum(1 for count in visit_counts.values() if count == 1)
    visited_multiple = sum(1 for count in visit_counts.values() if count > 1)
    
    # Prepare summary text
    summary = f"VISIT PATTERN ANALYSIS\n{'='*30}\n\n"
    summary += f"Total unique places visited: {total_unique_places}\n"
    summary += f"Total visits recorded: {total_visits}\n\n"
    
    summary += "PLACE VISIT FREQUENCY\n"
    summary += f"* Places visited just once: {visited_once} ({visited_once/total_unique_places*100:.1f}%)\n"
    summary += f"* Places visited multiple times: {visited_multiple} ({visited_multiple/total_unique_places*100:.1f}%)\n\n"
    
    summary += "VISIT EXPLORATION METRICS\n"
    summary += f"* New (exploration) visits: {new_visits} ({new_visits/total_visits*100:.1f}%)\n"
    summary += f"* Repeated visits: {repeated_visits} ({repeated_visits/total_visits*100:.1f}%)\n\n"
    
    # Add frequency distribution
    summary += "VISIT FREQUENCY DISTRIBUTION\n"
    for freq, count in sorted(frequency_counts.items()):
        summary += f"* {count} places visited {freq} times ({count/total_unique_places*100:.1f}%)\n"
    
    # Add daily visit breakdown if there are multiple dates
    dates = sorted(set(stays_df['date']))
    if len(dates) > 1:
        # Track seen places across days
        seen_places_cumulative = set()
        
        summary += "\nVISITS BY DATE\n"
        for date in dates:
            day_visits = stays_df[stays_df['date'] == date]
            day_places = set(day_visits['place_id'])
            
            # Count new vs. repeated places for this day
            new_places_today = day_places - seen_places_cumulative
            repeated_places_today = day_places & seen_places_cumulative
            
            summary += f"* {date}: {len(day_visits)} visits to {len(day_places)} places\n"
            summary += f"  - New places: {len(new_places_today)}, "
            summary += f"Repeated places: {len(repeated_places_today)}\n"
            
            # Update cumulative set
            seen_places_cumulative.update(new_places_today)
    
    return summary

# Generate a visit data table
def generate_visit_data_table(stays_df, visit_dates):
    """
    Generate a detailed table of visit data
    
    Args:
        stays_df: DataFrame with stay information
        visit_dates: Dictionary mapping place_ids to lists of dates
        
    Returns:
        DataFrame with visit metrics per place
    """
    if stays_df.empty:
        return pd.DataFrame()
    
    # Create place metrics
    place_metrics = []
    
    for place_id, dates in visit_dates.items():
        # Get all visits to this place
        place_visits = stays_df[stays_df['place_id'] == place_id]
        
        # Calculate metrics
        visit_count = len(dates)
        avg_duration = place_visits['duration_minutes'].mean()
        total_duration = place_visits['duration_minutes'].sum()
        
        # First and last visit
        if 'arrival' in place_visits.columns:
            first_visit = place_visits['arrival'].min()
            last_visit = place_visits['arrival'].max()
        else:
            first_visit = None
            last_visit = None
        
        # Calculate time span
        if first_visit and last_visit:
            time_span = (last_visit - first_visit).total_seconds() / 3600  # in hours
        else:
            time_span = 0
        
        place_metrics.append({
            'place_id': place_id,
            'visit_count': visit_count,
            'avg_duration_minutes': round(avg_duration, 1),
            'total_duration_minutes': round(total_duration, 1),
            'first_visit': first_visit,
            'last_visit': last_visit,
            'time_span_hours': round(time_span, 1),
            'latitude': place_visits['latitude'].mean(),
            'longitude': place_visits['longitude'].mean()
        })
    
    # Convert to DataFrame and sort by most visited
    places_df = pd.DataFrame(place_metrics).sort_values('visit_count', ascending=False)
    return places_df

# Main function to analyze GPS data
def analyze_gps_data(zip_path):
    """
    Analyze GPS data and produce required outputs
    
    Args:
        zip_path: Path to zip file containing GPX files
        
    Returns:
        Dictionary containing analysis results
    """
    print(f"Analyzing GPS data from {zip_path}...")
    
    # Parse the GPX files
    stays_df = parse_gpx_files_from_zip(zip_path)
    
    if stays_df.empty:
        print("No stays found in the data.")
        return {"error": "No stays found"}
    
    # Assign place IDs (group nearby stay locations)
    stays_with_places = assign_place_ids(stays_df)
    
    # Analyze visit patterns
    daily_stats, visit_dates = analyze_visit_patterns(stays_with_places)
    
    # Create transition graph
    transition_counts = create_transition_graph(stays_with_places)
    
    # Generate transition graph visualization
    plt = draw_transition_graph(stays_with_places, transition_counts)
    plt.savefig('place_transitions.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'place_transitions.png'")
    plt.close()
    
    # Generate visit data table
    visit_table = generate_visit_data_table(stays_with_places, visit_dates)
    visit_table.to_csv('visit_data.csv', index=False)
    print("Visit data table saved as 'visit_data.csv'")
    
    # Generate text summary
    summary = generate_visit_summary(stays_with_places, visit_dates)
    with open('visit_summary.txt', 'w') as f:
        f.write(summary)
    print("Visit summary saved as 'visit_summary.txt'")
    
    # Print the summary to console too
    print("\n" + summary)
    
    return {
        "stays_df": stays_with_places,
        "visit_dates": visit_dates,
        "transition_counts": transition_counts
    }

# Analyze visit patterns and count new vs. repeated visits
def analyze_visit_patterns(stays_df):
    """
    Analyze visit patterns, tracking new vs. repeated visits over time
    
    Args:
        stays_df: DataFrame with place_id and date information
        
    Returns:
        (daily_stats DataFrame, visit_dates dictionary)
    """
    if stays_df.empty:
        return pd.DataFrame(), {}

    visit_dates = {}
    daily_stats = []
    
    # Keep track of seen places (for true first visits)
    seen_places = set()
    
    # Process each day
    for date, group in stays_df.groupby('date'):
        places_today = group['place_id'].tolist()
        unique_places_today = set(places_today)
        
        # Count new vs repeated places
        new_places = unique_places_today - seen_places
        repeated_places = unique_places_today & seen_places
        
        # Count actual visits
        new_visits = 0
        repeated_visits = 0
        
        # Process each visit
        for place_id in places_today:
            if place_id in visit_dates:
                # Already seen this place before
                visit_dates[place_id].append(date)
                repeated_visits += 1
            else:
                # First time seeing this place
                visit_dates[place_id] = [date]
                new_visits += 1
        
        # Add to daily stats
        daily_stats.append({
            'date': date,
            'unique_places': len(unique_places_today),
            'total_visits': len(places_today),
            'new_visits': new_visits,
            'repeated_visits': repeated_visits
        })
        
        # Update seen places for next iteration
        seen_places.update(new_places)

    return pd.DataFrame(daily_stats), visit_dates


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use command line argument as file path
        zip_path = sys.argv[1]
    else:
        # Default path if none provided
        zip_path = "gps_data.zip"  # Generic filename as an example
    
    print(f"Processing file: {zip_path}")
    analyze_gps_data(zip_path)