import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """加载数据"""
    try:
        data = pd.read_excel(file_path)
        X = data.iloc[:,:64].to_numpy()
        y = data.iloc[:,64].to_numpy()
        return X, y
    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def process_data(X, y, test_size=0.2, random_state=42):
    """分割数据集"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """分割数据集并标准化"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def stand(X):
    """标准化"""
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def reshape(X):
    """重塑数据为图像"""
    X = X.reshape(-1, 8, 8, 1)
    return X