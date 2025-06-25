import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TreeNode:
    feature: Optional[str] = None
    value: Optional[float] = None
    children: Dict[Any, 'TreeNode'] = field(default_factory=dict)
    label: Optional[Any] = None

    def __repr__(self):
        if self.label is not None:
            return f"Leaf(label={self.label})"
        return f"Node(feature={self.feature}, value={self.value})"
    
    
    
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.default_class = None  # Store most common class as fallback
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame, np.ndarray]) -> 'DecisionTreeClassifier':
        """Fit the decision tree model to the training data.
        This method builds the decision tree based on the provided features and target labels.

        Args:
            X (pandas.DataFrame): The feature dataset with shape (n_samples, n_features)
            y (pandas.Series or numpy.ndarray): Target labels for classification

        Returns:
            self: Returns the instance itself
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")        
        if not isinstance(y, (pd.Series, np.ndarray, pd.DataFrame)):
            raise ValueError("y must be a pandas Series or numpy ndarray.")
        
        # Ensure y is a Series for consistency
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
          # Store the most common class as fallback for missing predictions
        if isinstance(y, pd.DataFrame):
            self.default_class = y.iloc[:, 0].mode().iloc[0]
            self.classes_ = sorted(y.iloc[:, 0].unique())
        else:
            self.default_class = y.mode().iloc[0]
            self.classes_ = sorted(y.unique())
        
        self.tree = self._build_tree(X, y)
        return self
        
    def _predict_row(self, node: TreeNode, row: pd.Series) -> Any:
        """
        Predict the class label for a single row of features using the decision tree.
        This method traverses the tree based on the feature values in the row.

        Args:
            node (TreeNode): The current node in the decision tree
            row (pandas.Series): A single row of features

        Returns:
            Any: The predicted class label for the row
        """
        if node.label is not None:
            # print(f"DEBUG: Found label {node.label}")
            return node.label
        
        feature_value = row[node.feature]
        
        # Handle missing values
        if pd.isna(feature_value) or feature_value is None:
            # If feature value is missing, we can't make a decision
            # Return the most common class from available children or None
            if node.children:
                # Try to find a default path or return None to be handled upstream
                return None
            else:
                return None
        
        if node.value is not None:
            # For continuous features
            if feature_value <= node.value:
                child_node = node.children.get('left')
            else:
                child_node = node.children.get('right')
        else:
            # For categorical features
            child_node = node.children.get(feature_value)
        
        if child_node is None:
            return None
        
        return self._predict_row(child_node, row)    
    
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict the class labels for the provided features using the trained decision tree.
        Args:
            X (pandas.DataFrame or numpy.ndarray): The feature dataset for which to predict labels
        Returns:
            numpy.ndarray: Predicted class labels for each sample in X
        """
        if self.tree is None:
            raise ValueError("The model has not been trained yet. Call fit() before predict().")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        predictions = []
        for _, row in X.iterrows():
            prediction = self._predict_row(self.tree, row)
            # If prediction is None (due to missing values or incomplete tree paths),
            # use the most common class from training data as fallback
            if prediction is None:
                prediction = self.default_class
            predictions.append(prediction)
        
        return np.array(predictions)
        
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for the provided features using the trained decision tree.
        For decision trees, this returns the class distribution at leaf nodes.
        
        Args:
            X (pandas.DataFrame or numpy.ndarray): The feature dataset for which to predict probabilities
            
        Returns:
            numpy.ndarray: Predicted class probabilities with shape (n_samples, n_classes)
        """
        if self.tree is None:
            raise ValueError("The model has not been trained yet. Call fit() before predict_proba().")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        # Get unique classes from training data
        if hasattr(self, 'classes_'):
            classes = self.classes_
        else:
            # If classes_ not stored, use default_class as fallback
            classes = [self.default_class]
        
        probabilities = []
        for _, row in X.iterrows():
            prediction = self._predict_row(self.tree, row)
            # For decision trees, we create a one-hot encoded probability
            # In a more sophisticated implementation, you could store class distributions at leaf nodes
            if prediction is None:
                prediction = self.default_class
            
            # Create probability array with 1.0 for predicted class, 0.0 for others
            prob_row = np.zeros(len(classes))
            if prediction in classes:
                class_idx = classes.index(prediction)
                prob_row[class_idx] = 1.0
            else:
                # If prediction not in known classes, use default
                if self.default_class in classes:
                    default_idx = classes.index(self.default_class)
                    prob_row[default_idx] = 1.0
                else:
                    prob_row[0] = 1.0  # Fallback to first class
            
            probabilities.append(prob_row)
        
        return np.array(probabilities)

    def _entropy(self, y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate the entropy of the target labels.
        Entropy is a measure of the uncertainty in the data.

        Args:
            y (Union[pd.Series, np.ndarray]): The target labels

        Returns:
            float: The entropy of the target labels
        """
        if isinstance(y, pd.Series):
            y = y.value_counts(normalize=True)
        else:
            y = pd.Series(y).value_counts(normalize=True)
        
        return -np.sum(y * np.log2(y + 1e-9))
        
        
    def _information_gain(self, parent_y: Union[pd.Series, np.ndarray], *child_y: List[Union[pd.Series, np.ndarray]]) -> float:
        """
        Calculate the information gain from a parent node to its children nodes.
        This is a measure of how much information a feature provides about the class labels.

        Args:
            parent_y (Union[pd.Series, np.ndarray]): The target labels of the parent node
            *child_y (List[Union[pd.Series, np.ndarray]]): The target labels of the child nodes

        Returns:
            float: The information gain from the parent to the children nodes
        """
        # if len(parent_y) != sum(len(child) for child in child_y):
        #     raise ValueError("Parent and sum of children labels must have the same length.")
        
        parent_entropy = self._entropy(parent_y)
        child_entropy = sum((len(child) / len(parent_y)) * self._entropy(child) for child in child_y)
        
        return parent_entropy - child_entropy
            
        
    def _best_split_index_with_gain(self, X: pd.Series, y: Union[pd.Series, np.ndarray] ) -> Tuple[float, float]:
        """
        Determine the best split index for a given continuous feature.
        This method evaluates all possible split points for a continuous feature and selects the one that maximizes the information gain or minimizes the Gini index.

        Args:
            X (pd.Series): The feature values for a single feature column
            y (Union[pd.Series, np.ndarray]): The target labels

        Returns:
            float: The value of the best split point
            float: The maximum information gain or Gini index achieved by this split
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        
        # Handle missing values by dropping them
        valid_mask = pd.notnull(X)
        if not valid_mask.any():
            return 0.0, 0.0
        
        X_clean = X[valid_mask]
        if isinstance(y, pd.DataFrame):
            y_clean = y[valid_mask]
        elif isinstance(y, pd.Series):
            y_clean = y[valid_mask]
        else:
            y_clean = y[valid_mask.values]
        
        if len(X_clean) <= 1:
            return 0.0, 0.0
        
        sorted_indices = np.argsort(X_clean)
        X_sorted = X_clean.iloc[sorted_indices].values
        
        # Handle different types of y (DataFrame, Series, or numpy array)
        if isinstance(y_clean, pd.DataFrame):
            y_sorted = y_clean.iloc[sorted_indices].values.flatten()
        elif isinstance(y_clean, pd.Series):
            y_sorted = y_clean.iloc[sorted_indices].values
        else:
            y_sorted = y_clean[sorted_indices]
        
        # Calculate midpoints, ensuring we have valid numeric values
        midpoints = []
        for i in range(len(X_sorted) - 1):
            if X_sorted[i] != X_sorted[i + 1]:  # Only create midpoint if values are different
                midpoint = (X_sorted[i] + X_sorted[i + 1]) / 2
                midpoints.append(midpoint)
        
        if len(midpoints) == 0:
            return 0.0, 0.0
        
        midpoints = np.array(midpoints)
        
        gains = []
        for midpoint in midpoints:
            left_mask = X_sorted <= midpoint
            right_mask = X_sorted > midpoint
            
            if np.any(left_mask) and np.any(right_mask):
                gain = self._information_gain(y_sorted, y_sorted[left_mask], y_sorted[right_mask])
                gains.append(gain)
            else:
                gains.append(0.0)
        
        if len(gains) == 0:
            return 0.0, 0.0
        
        gains = np.array(gains)
        best_gain_index = np.argmax(gains)
        
        return midpoints[best_gain_index], np.max(gains)
    
    def is_continuous(self, X: pd.Series) -> bool:
        """
        Check if a feature is continuous based on its data type and unique values.
        A feature is considered continuous if it is numeric and has more than 0.1% unique values.

        Args:
            X (pd.Series): The feature values for a single feature column

        Returns:
            bool: True if the feature is continuous, False otherwise
        """
        return np.issubdtype(X.dtype, np.number) and (len(X.unique()) / len(X) > 0.001)
        
        
    def _best_split_feature(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Determine the best feature to split on based on information gain, gain ratio, or Gini index.
        This method evaluates all features and selects the one that maximizes the information gain or minimizes the Gini index.
        Args:
            X (pandas.DataFrame): The feature dataset
            y (Union[pd.Series, np.ndarray]): The target labels
        Returns:
            Any: The best feature to split on, including its name, values, series, type, and whether it is continuous.
        """
        if y is None or len(y) == 0:
            return None

        # Convert y to a consistent format (Series or array)
        if isinstance(y, pd.DataFrame):
            y_values = y.iloc[:, 0]  # Take the first column
        else:
            y_values = y

        features = [
            {
            "name": col,
            "values": X[col].unique(),
            "series": X[col],
            "type": X[col].dtype,
            # if the feature is continuous, it has more than 10% unique values and is numeric
            "is_continous": self.is_continuous(X[col]),
            "best_gain": 0,
            "best_value": None
            }
            for col in X.columns
        ]
        
        # print(f"Features: {features}")
        
        
        
        for feature in features:
            # print(f"Evaluating feature: {feature['name']}, is_continuous: {feature['is_continous']}")
            if feature["is_continous"]:
                # Sort the features for calculating midpoints
                sorted_indices = np.argsort(feature["series"])
                sorted_feature = feature["series"].iloc[sorted_indices]
                sorted_y = y_values.iloc[sorted_indices] if isinstance(y_values, pd.Series) else y_values[sorted_indices]
                
                best_value, best_gain = self._best_split_index_with_gain(sorted_feature, sorted_y)
                feature["best_value"] = best_value
                feature["best_gain"] = best_gain
            else:
                
                values_y = []
                for col in feature["values"]:
                    subset_y = y_values[feature["series"] == col]
                    if len(subset_y) == 0:
                        continue
                    values_y.append(subset_y)
                if len(values_y) == 0:
                    continue
                    
                gain = self._information_gain(y_values, *values_y)
                feature["best_index"] = None
                feature["best_value"] = None
                feature["best_gain"] = gain
                
        best_feature = max(features, key=lambda f: f["best_gain"])
        if best_feature["best_gain"] <= 0:
            return None
        
        return best_feature
    def _split_data(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame, np.ndarray], feature: Dict[str, Any]) -> List[Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame, np.ndarray]]]:
        """
        Split the dataset based on the best feature determined by _best_split_feature.
        This method creates subsets of the dataset for each unique value of the best feature.

        Args:
            X (pandas.DataFrame): The feature dataset
            y (Union[pd.Series, pd.DataFrame, np.ndarray]): The target labels
            feature (Dict[str, Any]): The best feature to split on

        Returns:
            List[Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame, np.ndarray]]]: A list of tuples containing the split datasets
        """
        if feature["is_continous"]:
            # Check if best_value is None to avoid comparison errors
            if feature["best_value"] is None:
                return []
            
            left_mask = X[feature["name"]] <= feature["best_value"]
            right_mask = X[feature["name"]] > feature["best_value"]
            
            left_X = X[left_mask]
            right_X = X[right_mask]
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            
            return [(left_X, left_y), (right_X, right_y)]
        else:
            subsets = []
            for value in feature["values"]:
                mask = X[feature["name"]] == value
                subset_X = X[mask]
                subset_y = y[mask]
                if len(subset_X) > 0:
                    subsets.append((subset_X, subset_y))
            return subsets
        


    def _build_tree(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame, np.ndarray], depth: int = 0) -> TreeNode:
        """
        Recursively build the decision tree based on the training data.
        This method creates a tree structure where each node represents a feature split, and leaf nodes represent class labels.

        Args:
            X (pd.DataFrame): The feature dataset with shape (n_samples, n_features)
            y (Union[pd.Series, pd.DataFrame, np.ndarray]): Target labels for classification
            depth (int, optional): Current depth of the tree. Defaults to 0.

        Returns:
            TreeNode: The decision tree root node
        """
        if len(y) == 0:
            return None
        
        # Check if all labels are the same
        if isinstance(y, pd.DataFrame):
            unique_values = y.iloc[:, 0].unique()
        elif isinstance(y, pd.Series):
            unique_values = y.unique()
        else:
            unique_values = np.unique(y)
            
        if len(unique_values) == 1:
            if isinstance(y, pd.DataFrame):
                return TreeNode(label=y.iloc[0, 0])
            elif isinstance(y, pd.Series):
                return TreeNode(label=y.iloc[0])
            else:
                return TreeNode(label=y[0])
        
        # Check maximum depth
        if self.max_depth is not None and depth >= self.max_depth:
            if isinstance(y, pd.DataFrame):
                return TreeNode(label=y.iloc[:, 0].mode().iloc[0])
            elif isinstance(y, pd.Series):
                return TreeNode(label=y.mode().iloc[0])
            else:
                values, counts = np.unique(y, return_counts=True)
                return TreeNode(label=values[np.argmax(counts)])
        
        # Find the best feature to split on
        best_feature = self._best_split_feature(X, y)
        if best_feature is None:
            if isinstance(y, pd.DataFrame):
                return TreeNode(label=y.iloc[:, 0].mode().iloc[0])
            elif isinstance(y, pd.Series):
                return TreeNode(label=y.mode().iloc[0])
            else:
                values, counts = np.unique(y, return_counts=True)
                return TreeNode(label=values[np.argmax(counts)])
        
        # print(f"Best feature: {best_feature['name']}, Gain: {best_feature['best_gain']}, Value: {best_feature['best_value']}")
        
        # Split the data based on the best feature
        subsets = self._split_data(X, y, best_feature)
        
        if len(subsets) == 0:
            if isinstance(y, pd.DataFrame):
                return TreeNode(label=y.iloc[:, 0].mode().iloc[0])
            elif isinstance(y, pd.Series):
                return TreeNode(label=y.mode().iloc[0])
            else:
                values, counts = np.unique(y, return_counts=True)
                return TreeNode(label=values[np.argmax(counts)])
        
        # print(f"Subsets created: {len(subsets)}")
        
        # Create the tree node
        node = TreeNode(feature=best_feature["name"], value=best_feature["best_value"])
        
        if best_feature["is_continous"]:
            # For continuous features, we have left (≤) and right (>) branches
            left_X, left_y = subsets[0]
            right_X, right_y = subsets[1]
            
            left_node = self._build_tree(left_X, left_y, depth + 1)
            if left_node is not None:
                node.children["left"] = left_node
                
            right_node = self._build_tree(right_X, right_y, depth + 1)
            if right_node is not None:
                node.children["right"] = right_node
        else:
            # For categorical features, we have one branch per value
            for (subset_X, subset_y), value in zip(subsets, best_feature["values"]):
                if len(subset_X) == 0:
                    continue
                child_node = self._build_tree(subset_X, subset_y, depth + 1)
                if child_node is not None:
                    node.children[value] = child_node
        
        return node
        
    def __repr__(self):
        return f"DecisionTree(max_depth={self.max_depth}, tree={self.tree})"