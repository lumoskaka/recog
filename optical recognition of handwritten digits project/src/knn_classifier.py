import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

class KNNClassifier:
    def __init__(self, n_neighbors=6):
        """初始化 KNN 分类器"""
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
    
    def fit(self, X_train, y_train):
        """训练模型"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """预测结果"""
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy
