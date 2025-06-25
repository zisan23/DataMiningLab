import numpy as np
import pandas as pd
from typing import  Dict, Any, Union, Tuple


class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1):
        self.alpha = alpha
        
    def _calculate_class_priors(self, y: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[Any, float]:
        """
        Calculate the prior probabilities of each class.
        
        Parameters:
        y (Union[pd.DataFrame, pd.Series, np.ndarray]): The target variable.
        
        Returns:
        Dict[Any, float]: A dictionary mapping each class to its prior probability.
        """
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        class_counts = y.value_counts()
        total_count = len(y)
        
        return {
            cls: (count+self.alpha) / (total_count + self.alpha * len(class_counts))
            for cls, count in class_counts.items()
        }
        
    def _is_continous(self, X: pd.Series) -> bool:
        """
        Check if a feature is continuous based on its data type and unique values.
        A feature is considered continuous if it is numeric and has more than 10% unique values.

        Args:
            X (pd.Series): The feature values for a single feature column

        Returns:
            bool: True if the feature is continuous, False otherwise
        """
        return np.issubdtype(X.dtype, np.number) and (len(X.unique()) / len(X) > 0.001)
    
    
    def _calculate_discrete_feature_conditional_probabilities(self, X: pd.Series, y: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[Any, Dict[str, float]]:
        """
        Calculate the conditional probabilities of discrete features given each class.
        
        Parameters:
        X (pd.Series): The feature values for a single feature column.
        y (Union[pd.DataFrame, pd.Series, np.ndarray]): The target variable.
        
        Returns:
        Dict[Any, Dict[str, float]]: A dictionary mapping each class to its conditional probabilities for the feature.
        """
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        class_conditional_probs = {}
        class_counts = y.value_counts()
        
        for cls in class_counts.index:
            class_data = X[y == cls]
            feature_counts = class_data.value_counts()
            total_feature_count = len(class_data)
            
            class_conditional_probs[cls] = {
                X.name: {
                    value: {
                        "prob":(count + self.alpha) / (total_feature_count + self.alpha * len(feature_counts)),
                    }
                    for value, count in feature_counts.items()
                }
            }
            
            class_conditional_probs[cls][X.name]["is_continuous"] = False
            class_conditional_probs[cls][X.name]["mean"] = None
            class_conditional_probs[cls][X.name]["std"] = None
            
            
            
            
            
        return class_conditional_probs
    
    
    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """
        Calculate the probability density function of a Gaussian distribution.
        
        Parameters:
        x (float): The value for which to calculate the PDF.
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
        
        Returns:
        float: The probability density at x.
        """
        if std == 0:
            return 0.0
        coeff = 1 / (std * np.sqrt(2 * np.pi))
        exponent = -((x - mean) ** 2) / (2 * std ** 2)
        return coeff * np.exp(exponent)
    
    
    def _calculate_continuous_feature_conditional_probabilities(self, X: pd.Series, y: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[Any, Tuple[float, float]]:
        """
        Calculate the conditional probabilities of continuous features given each class.
        
        Parameters:
        X (pd.Series): The feature values for a single feature column.
        y (Union[pd.DataFrame, pd.Series, np.ndarray]): The target variable.
        
        Returns:
        Dict[Any, Tuple[float, float]]: A dictionary mapping each class to its mean and standard deviation for the feature.
        """
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        class_conditional_probs = {}
        class_counts = y.value_counts()
        
        for cls in class_counts.index:
            class_data = X[y == cls]
            class_mean = class_data.mean()
            class_std = class_data.std(ddof=0)  # Use population standard deviation
            class_conditional_probs[cls] = {
                X.name: {
                    "mean": class_mean,
                    "std": class_std,
                    "is_continuous": True,                    
                }
            }
            
        return class_conditional_probs
            
        
    def _calculate_conditional_probabilities(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]) -> Dict[Any, Dict[str, float]]:
        """
        Calculate the conditional probabilities of features given each class.
        
        Parameters:
        X (pd.DataFrame): The feature set.
        y (Union[pd.DataFrame, pd.Series, np.ndarray]): The target variable.
        
        Returns:
        Dict[Any, Dict[str, float]]: A dictionary mapping each class to its conditional probabilities for each feature.
        """
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        class_conditional_probs = {}
        class_counts = y.value_counts()
        
        for cls in class_counts.index:
            class_data: pd.DataFrame = X[y == cls]
            
            for feature in class_data.columns:
                
                if self._is_continous(X[feature]):
                    # print(f"Feature {feature} is continuous.")
                    feature_probs = self._calculate_continuous_feature_conditional_probabilities(class_data[feature], y)
                else:
                    # print(f"Feature {feature} is discrete.")
                    feature_probs = self._calculate_discrete_feature_conditional_probabilities(class_data[feature], y)
                
                if cls not in class_conditional_probs:
                    class_conditional_probs[cls] = {}
                    
                class_conditional_probs[cls].update(feature_probs[cls])
        return class_conditional_probs
                
        
        
    
    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]) -> 'NaiveBayesClassifier':
        """
        Fit the Naive Bayes classifier to the training data.
        
        Parameters:
        X (pd.DataFrame): The feature set.
        y (Union[pd.DataFrame, pd.Series, np.ndarray]): The target variable.
        
        Returns:
        NaiveBayesClassifier: The fitted classifier.
        """
        self.class_priors = self._calculate_class_priors(y)
        self.conditional_probs = self._calculate_conditional_probabilities(X, y)
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the given feature set.
        
        Parameters:
        X (pd.DataFrame): The feature set for which to predict class probabilities.
        
        Returns:
        np.ndarray: An array of class probabilities with shape (n_samples, n_classes).
        """
        predictions = [{
            cls: self.class_priors[cls] for cls in self.class_priors
        }
        for _ in range(len(X))]
        
        for i, (_, row) in enumerate(X.iterrows()):
            for cls in self.class_priors:
                class_prob = self.class_priors[cls]
                
                for feature, value in row.items():
                    if feature in self.conditional_probs[cls]:
                        feature_probs = self.conditional_probs[cls][feature]
                        
                        if feature_probs["is_continuous"]:
                            mean = feature_probs["mean"]
                            std = feature_probs["std"]
                            class_prob *= self._gaussian_pdf(value, mean, std)
                        else:
                            if value in feature_probs:
                                class_prob *= feature_probs[value]["prob"]
                            else:
                                class_prob *= 0
                                
                predictions[i][cls] = class_prob
        
        # Convert to numpy array with consistent class ordering
        classes = sorted(self.class_priors.keys())
        probabilities = np.array([[pred[cls] for cls in classes] for pred in predictions])
        
        # Normalize probabilities to sum to 1 for each sample
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        probabilities = probabilities / row_sums
        
        return probabilities    
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the class labels for the given feature set.
        
        Parameters:
        X (pd.DataFrame): The feature set for which to predict class labels.
        
        Returns:
        np.ndarray: An array of predicted class labels.
        """
        # Get probabilities and return the class with highest probability
        probabilities = self.predict_proba(X)
        classes = sorted(self.class_priors.keys())
        predicted_indices = np.argmax(probabilities, axis=1)
        return np.array([classes[idx] for idx in predicted_indices])
            
        
    def __repr__(self):
        return f"NaiveBayesClassifier(alpha={self.alpha})"

