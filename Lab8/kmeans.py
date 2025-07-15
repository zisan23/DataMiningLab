import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')


def extract_numerical_features(dataset):   
    """Extract numerical features from the dataset for clustering."""
    
    features_dataframe = dataset.data.features
    
    categorical_features = []
    numerical_features = []
    
    for column in features_dataframe.columns:
        data_type = features_dataframe[column].dtype
        
        if data_type == 'object' or data_type == 'string' or data_type == 'bool':
            categorical_features.append(column)
        elif hasattr(data_type, 'name') and 'category' in str(data_type):
            categorical_features.append(column)
        else:
            numerical_features.append(column)
    
    if categorical_features:
        print(f"\n Categorical features detected:")
        for feature in categorical_features:
            print(f"   - {feature}")
        print(f"\n Keeping {len(numerical_features)} numerical features for clustering:")
        for feature in numerical_features:
            print(f"   - {feature}")
        
        filtered_features_dataframe = features_dataframe[numerical_features]
        
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
    
    print("Data preprocessing completed.")
    
    return normalized_data, scaler, valid_indices


class KMeans:
    """Implementation of K-Means clustering algorithm."""
    
    def __init__(self, n_clusters, max_iterations=100, convergence_threshold=1e-4, initialization='kmeans++', random_state=42):
        """Initialize the KMeans algorithm with specified parameters."""
        
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.initialization = initialization
        self.random_state = random_state
        
        # Results will be stored here after fitting
        self.cluster_centers = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        
    def _initialize_centroids(self, data_points):
        """Initialize cluster centroids based on the specified method."""
        
        np.random.seed(self.random_state)
        n_samples, n_features = data_points.shape
        
        if self.initialization == 'random':
            # Random initialization within data bounds
            min_vals = data_points.min(axis=0)
            max_vals = data_points.max(axis=0)
            centroids = np.random.uniform(min_vals, max_vals, (self.n_clusters, n_features))
        elif self.initialization == 'kmeans++':
            # K-means++ initialization
            centroids = np.zeros((self.n_clusters, n_features))
            
            # Choose first centroid randomly
            centroids[0] = data_points[np.random.randint(n_samples)]
            
            # Choose remaining centroids
            for i in range(1, self.n_clusters):
                # Calculate distances from each point to nearest centroid
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids[:i]]) for x in data_points])
                
                # Choose next centroid with probability proportional to squared distance
                probabilities = distances / distances.sum()
                cumulative_probabilities = probabilities.cumsum()
                random_value = np.random.rand()
                
                for j, probability in enumerate(cumulative_probabilities):
                    if random_value < probability:
                        centroids[i] = data_points[j]
                        break
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")
        
        return centroids
    
    def _assign_clusters(self, data_points, centroids):
        """Assign each data point to the nearest centroid."""
        
        n_samples = data_points.shape[0]
        cluster_labels = np.zeros(n_samples)
        
        for i, point in enumerate(data_points):
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_labels[i] = np.argmin(distances)
        
        return cluster_labels.astype(int)
    
    def _update_centroids(self, data_points, cluster_labels):
        """Recalculate centroids based on the current cluster assignments."""
        
        n_features = data_points.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for i in range(self.n_clusters):
            cluster_points = data_points[cluster_labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # If cluster is empty, keep the old centroid
                new_centroids[i] = self.cluster_centers[i] if self.cluster_centers is not None else np.random.rand(n_features)
        
        return new_centroids
    
    def _calculate_inertia(self, data_points, cluster_labels, centroids):
        """Calculate the sum of squared distances from points to their centroids."""
        
        total_inertia = 0
        for i in range(self.n_clusters):
            cluster_points = data_points[cluster_labels == i]
            if len(cluster_points) > 0:
                total_inertia += np.sum((cluster_points - centroids[i]) ** 2)
        return total_inertia
    
    def fit(self, data_points):
        """Fit the KMeans model to the provided data."""
        
        # Initialize centroids
        self.cluster_centers = self._initialize_centroids(data_points)
        
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            cluster_labels = self._assign_clusters(data_points, self.cluster_centers)
            
            # Update centroids
            new_centroids = self._update_centroids(data_points, cluster_labels)
            
            # Check for convergence
            centroid_movement = np.linalg.norm(new_centroids - self.cluster_centers)
            
            self.cluster_centers = new_centroids
            
            if centroid_movement < self.convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                self.n_iter_ = iteration + 1
                break
        else:
            print(f"Reached maximum iterations ({self.max_iterations})")
            self.n_iter_ = self.max_iterations
        
        # Store final results
        self.labels_ = self._assign_clusters(data_points, self.cluster_centers)
        self.inertia_ = self._calculate_inertia(data_points, self.labels_, self.cluster_centers)
        
        return self
    
    def predict(self, data_points):
        """Predict the closest cluster for each sample in data_points."""
        
        if self.cluster_centers is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._assign_clusters(data_points, self.cluster_centers)
    
    def fit_predict(self, data_points):
        """Fit the model and predict cluster labels in one step."""
        
        self.fit(data_points)
        return self.labels_


def calculate_inertia_values(data_points, max_k=10):
    """Compute inertia values for different numbers of clusters."""
    
    print(f"Finding optimal number of clusters (k=1 to {max_k})...")
    
    inertia_values = []
    
    for k in range(1, max_k + 1):
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_points)
        inertia_values.append(kmeans.inertia_)
    
    return inertia_values


def find_optimal_k_elbow(inertia_values):
    """Find the optimal number of clusters using the elbow method."""
    
    inertia_values = np.array(inertia_values)
    n_points = len(inertia_values)
    
    x_values = np.arange(1, n_points + 1)
    y_values = inertia_values
    
    first_point = np.array([x_values[0], y_values[0]])
    last_point = np.array([x_values[-1], y_values[-1]])
    
    max_distance = 0
    optimal_k = 1
    
    for i in range(len(x_values)):
        current_point = np.array([x_values[i], y_values[i]])
        line_vector = last_point - first_point
        point_vector = current_point - first_point
        line_length = np.linalg.norm(line_vector)
        
        if line_length > 0:
            line_unit_vector = line_vector / line_length
            projection_length = np.dot(point_vector, line_unit_vector)
            projection = projection_length * line_unit_vector
            distance = np.linalg.norm(point_vector - projection)
            
            if distance > max_distance:
                max_distance = distance
                optimal_k = i + 1
    
    return optimal_k


def calculate_silhouette_scores(data_points, cluster_labels):
    """Calculate silhouette scores for clustering evaluation."""

    n_samples = data_points.shape[0]
    n_clusters = len(np.unique(cluster_labels))
    
    if n_clusters == 1:
        return 0.0, np.zeros(n_samples)
    
    silhouette_values = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Current point and its cluster
        point = data_points[i]
        own_cluster = cluster_labels[i]
        
        # Points in same cluster
        same_cluster_points = data_points[cluster_labels == own_cluster]
        if len(same_cluster_points) > 1:
            # Calculate mean distance to points in same cluster (excluding self)
            same_cluster_distances = [np.linalg.norm(point - other) for j, other in enumerate(same_cluster_points) if j != np.where(cluster_labels == own_cluster)[0].tolist().index(i)]
            a_i = np.mean(same_cluster_distances) if same_cluster_distances else 0
        else:
            a_i = 0
        
        # Calculate mean distance to points in each other cluster
        b_i = float('inf')
        for cluster_id in np.unique(cluster_labels):
            if cluster_id != own_cluster:
                other_cluster_points = data_points[cluster_labels == cluster_id]
                if len(other_cluster_points) > 0:
                    other_cluster_distances = [np.linalg.norm(point - other) for other in other_cluster_points]
                    mean_distance = np.mean(other_cluster_distances)
                    b_i = min(b_i, mean_distance)
        
        # Calculate silhouette score
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0
    
    silhouette_avg = np.mean(silhouette_values)
    return silhouette_avg, silhouette_values
