from src.knn_classifier import KNNClassifier
from src.utils import load_data, process_data
import numpy as np

# 加载真实数据
data_path = 'data/optdigits-tra.xlsx'  
try:
    X, y = load_data(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 数据预处理
X_train, X_test, y_train, y_test = process_data(X, y)

data_path = 'data/optdigits-tes.xlsx'  
    X_test_outer, y_test_outer = load_data(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# 初始化 KNN 模型
knn = KNNClassifier(n_neighbors=6)

# 训练 KNN 模型
print("Training KNN model...")
knn.fit(X_train, y_train)

# 测试并评估模型
print("Evaluating KNN model-written...")
accuracy = knn.evaluate(X_test, y_test)
print(f"KNN Model Accuracy on Test Data: {accuracy:.3f}")

# 测试单个样本
sample_idx = 0  
sample = X_test[sample_idx].reshape(1, -1)  
predicted_label = knn.predict(sample)
true_label = y_test[sample_idx]
print(f"Predicted Label: {predicted_label[0]}, True Label: {true_label}")

# 测试并评估模型
print("Evaluating KNN model-nonwritten...")
accuracy = knn.evaluate(X_test_outer, y_test_outer)
print(f"KNN Model Accuracy on Test Data: {accuracy:.3f}")

# 测试单个样本
sample_idx = 0  
sample = X_test[sample_idx].reshape(1, -1)  
predicted_label = knn.predict(sample)
true_label = y_test[sample_idx]
print(f"Predicted Label: {predicted_label[0]}, True Label: {true_label}")

