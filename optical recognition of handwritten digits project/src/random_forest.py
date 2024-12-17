import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

class RandomForestClassifierWrapper:
    def __init__(self, n_estimators=100, random_state=42,class_weight='balanced'):
        """初始化随机森林分类器"""
        self.model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced')

    
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
