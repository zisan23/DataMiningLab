import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def extract_numerical_features(dataset):
    """Extract numerical features from the dataset for clustering."""
    
    # Get feature information
    features_dataframe = dataset.data.features
    
    # Check data types and identify categorical features
    categorical_features = []
    numerical_features = []
    
    for column in features_dataframe.columns:
        data_type = features_dataframe[column].dtype
        
        # Check if the column is categorical (object, string, or boolean)
        if data_type == 'object' or data_type == 'string' or data_type == 'bool':
            categorical_features.append(column)
        elif hasattr(data_type, 'name') and 'category' in str(data_type):
            categorical_features.append(column)
        else:
            numerical_features.append(column)
    
    if categorical_features:
        print(f"\n  Categorical features detected and will be removed:")
        for feature in categorical_features:
            print(f"   - {feature}")
        print(f"\n Keeping {len(numerical_features)} numerical features for clustering:")
        for feature in numerical_features:
            print(f"   - {feature}")
        
        # Filter the dataset to keep only numerical features
        filtered_features_dataframe = features_dataframe[numerical_features]
        
        # Create a filtered dataset object
        filtered_dataset = type(dataset)()
        filtered_dataset.data = type(dataset.data)()
        filtered_dataset.data.features = filtered_features_dataframe
        filtered_dataset.data.targets = dataset.data.targets
        
        return filtered_dataset, categorical_features
    
    return dataset, []


def preprocess_data(data_matrix):
    """Clean missing values and normalize the data for clustering."""
    
    # Check for missing values
    if np.isnan(data_matrix).any():
        # Remove rows with any missing values
        valid_indices = ~np.isnan(data_matrix).any(axis=1)
        cleaned_data = data_matrix[valid_indices]
    else:
        # No missing values found
        cleaned_data = data_matrix
        valid_indices = np.ones(data_matrix.shape[0], dtype=bool)
    
    # Standardize features
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(cleaned_data)
    
    return normalized_data, scaler, valid_indices


class DBSCAN:
    """Implementation of DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm."""
    
    def __init__(self, epsilon=0.5, min_samples=5):
        """Initialize the DBSCAN algorithm with specified parameters.
        
        Args:
            epsilon: The maximum distance between two samples for them to be considered neighbors
            min_samples: The number of samples in a neighborhood for a point to be considered a core point
        """
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.cluster_labels = None
        self.core_samples = None
        self.num_clusters = None
        self.noise_count = None
        
    def _find_neighbors(self, data_matrix, point_idx):
        """Find all neighbors within epsilon distance of the given point."""
        
        neighbors = []
        current_point = data_matrix[point_idx]
        
        for i, other_point in enumerate(data_matrix):
            distance = np.linalg.norm(current_point - other_point)
            if distance <= self.epsilon:
                neighbors.append(i)
        
        return neighbors
    
    def _expand_cluster(self, data_matrix, point_idx, neighbor_indices, cluster_id, visited_points, cluster_labels):
        """Expand the cluster from a core point to include all density-connected points."""
        
        # Add point to cluster
        cluster_labels[point_idx] = cluster_id
        
        # Process each neighbor
        i = 0
        while i < len(neighbor_indices):
            neighbor_idx = neighbor_indices[i]
            
            # If neighbor hasn't been visited, mark as visited and get its neighbors
            if neighbor_idx not in visited_points:
                visited_points.add(neighbor_idx)
                neighbor_neighbors = self._find_neighbors(data_matrix, neighbor_idx)
                
                # If neighbor is also a core point, add its neighbors to the current cluster
                if len(neighbor_neighbors) >= self.min_samples:
                    # Add new neighbors to the list (union operation)
                    for nn in neighbor_neighbors:
                        if nn not in neighbor_indices:
                            neighbor_indices.append(nn)
            
            # If neighbor is not yet assigned to any cluster, assign it to current cluster
            if cluster_labels[neighbor_idx] == -1:  # -1 means unassigned
                cluster_labels[neighbor_idx] = cluster_id
            
            i += 1
    
    def fit_predict(self, data_matrix):
        """Fit the DBSCAN clustering model and return cluster labels."""
        
        n_samples = data_matrix.shape[0]
        
        # Initialize labels (-1 means unassigned/noise)
        cluster_labels = np.full(n_samples, -1)
        visited_points = set()
        cluster_id = 0
        core_samples = []
        
        for idx in range(n_samples):
            # Skip if already visited
            if idx in visited_points:
                continue
            
            visited_points.add(idx)
            
            # Get neighbors within epsilon distance
            neighbor_indices = self._find_neighbors(data_matrix, idx)
            
            # Check if point is a core point
            if len(neighbor_indices) >= self.min_samples:
                core_samples.append(idx)
                # Expand cluster from this core point
                self._expand_cluster(data_matrix, idx, neighbor_indices, cluster_id, visited_points, cluster_labels)
                cluster_id += 1
        
        # Store results
        self.cluster_labels = cluster_labels
        self.core_samples = np.array(core_samples)
        self.num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        self.noise_count = list(cluster_labels).count(-1)
        
        return cluster_labels
    
    def fit(self, data_matrix):
        """Fit the DBSCAN clustering model to the provided data."""
        
        self.fit_predict(data_matrix)
        return self


def calculate_silhouette_score(data_matrix, cluster_labels):
    """Calculate the silhouette score for clustering evaluation."""
    
    # Remove noise points (label = -1) for silhouette calculation
    non_noise = cluster_labels != -1
    if np.sum(non_noise) == 0:
        return 0.0
    
    filtered_data = data_matrix[non_noise]
    filtered_labels = cluster_labels[non_noise]
    
    n_clusters = len(np.unique(filtered_labels))
    if n_clusters <= 1:
        return 0.0
    
    n_samples = filtered_data.shape[0]
    silhouette_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Current point and its cluster
        point = filtered_data[i]
        own_cluster = filtered_labels[i]
        
        # Points in same cluster
        same_cluster_points = filtered_data[filtered_labels == own_cluster]
        if len(same_cluster_points) > 1:
            # Calculate mean distance to points in same cluster (excluding self)
            same_cluster_distances = []
            for j, other in enumerate(same_cluster_points):
                # Skip the point itself
                original_indices = np.where(filtered_labels == own_cluster)[0]
                if original_indices[j] != i:
                    same_cluster_distances.append(np.linalg.norm(point - other))
            a_i = np.mean(same_cluster_distances) if same_cluster_distances else 0
        else:
            a_i = 0
        
        # Calculate mean distance to points in each other cluster
        b_i = float('inf')
        for cluster_id in np.unique(filtered_labels):
            if cluster_id != own_cluster:
                other_cluster_points = filtered_data[filtered_labels == cluster_id]
                if len(other_cluster_points) > 0:
                    other_cluster_distances = [np.linalg.norm(point - other) for other in other_cluster_points]
                    mean_distance = np.mean(other_cluster_distances)
                    b_i = min(b_i, mean_distance)
        
        # Calculate silhouette score
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0
    
    return np.mean(silhouette_values)


def optimize_parameters(data_matrix, epsilon_range=None, min_samples_range=None, max_combinations=50):
    """Find optimal DBSCAN parameters based on silhouette score and noise ratio."""
    
    if epsilon_range is None:
        # Compute k-distance graph for epsilon range selection
        print("  - Computing k-distance graph for epsilon range selection...")
        k = min(10, max(4, int(np.log(len(data_matrix)))))
        distances = []
        for point in data_matrix:
            point_distances = [np.linalg.norm(point - other) for other in data_matrix]
            point_distances.sort()
            distances.append(point_distances[k])
        
        distances.sort()
        epsilon_min = distances[len(distances) // 4]
        epsilon_max = distances[3 * len(distances) // 4]
        epsilon_range = np.linspace(epsilon_min, epsilon_max, 8)
        print(f"  - epsilon range: {epsilon_min:.3f} to {epsilon_max:.3f}")
    
    if min_samples_range is None:
        # min_samples typically ranges from 2 to 2*dimensions
        n_features = data_matrix.shape[1]
        min_samples_min = max(2, n_features // 2)
        min_samples_max = min(2 * n_features, max(6, data_matrix.shape[0] // 20))
        min_samples_range = range(min_samples_min, min_samples_max + 1)
        print(f"  - min_samples range: {min_samples_min} to {min_samples_max}")
    
    # Limit combinations for efficiency
    all_combinations = list(product(epsilon_range, min_samples_range))
    if len(all_combinations) > max_combinations:
        step = len(all_combinations) // max_combinations
        parameter_combinations = all_combinations[::step][:max_combinations]
    else:
        parameter_combinations = all_combinations
    
    parameter_results = []
    
    for i, (epsilon, min_samples) in enumerate(parameter_combinations):
        print(f"Testing combination {i+1}/{len(parameter_combinations)}: epsilon={epsilon:.3f}, min_samples={min_samples}")
        
        dbscan = DBSCAN(epsilon=epsilon, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(data_matrix)
        
        n_clusters = dbscan.num_clusters
        n_noise = dbscan.noise_count
        noise_ratio = n_noise / len(data_matrix) if len(data_matrix) > 0 else 1.0
        
        silhouette_score = calculate_silhouette_score(data_matrix, cluster_labels)
        
        # Combined score: weight silhouette score and penalize too much noise
        if n_clusters == 0:
            combined_score = 0
        else:
            combined_score = silhouette_score * (1 - noise_ratio * 0.5)
        
        parameter_results.append({
            'epsilon': epsilon,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette_score': silhouette_score,
            'combined_score': combined_score
        })
    
    # Sort by combined score
    parameter_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Print top 5 results for debugging
    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(parameter_results[:5]):
        print(f"  {i+1}. epsilon={result['epsilon']:.3f}, min_samples={result['min_samples']}, "
              f"clusters={result['n_clusters']}, noise={result['noise_ratio']:.3f}, "
              f"silhouette={result['silhouette_score']:.3f}, score={result['combined_score']:.3f}")
    
    return parameter_results

