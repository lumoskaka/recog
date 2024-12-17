from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
try:
    X_test_outer, y_test_outer = load_data(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 初始化随机森林模型
rf = RandomForestClassifier(n_estimators=100, random_state=42)  

# 训练模型
print("Training Random Forest model...")
rf.fit(X_train, y_train)

# 在测试集上评估模型
print("Evaluating Random Forest model-written...")
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy on Test Data: {accuracy:.3f}")

# 测试单个样本
sample_idx = 0  
sample = X_test[sample_idx].reshape(1, -1)  
predicted_label = rf.predict(sample)
true_label = y_test[sample_idx]
print(f"Predicted Label: {predicted_label[0]}, True Label: {true_label}")

# 在测试集上评估模型
print("Evaluating Random Forest model-nonwritten...")
y_pred_outer = rf.predict(X_test_outer)
accuracy = accuracy_score(y_test_outer, y_pred_outer)
print(f"Random Forest Model Accuracy on Test Data: {accuracy:.3f}")

# 测试单个样本
sample_idx = 0  
sample = X_test[sample_idx].reshape(1, -1)  
predicted_label = rf.predict(sample)
true_label = y_test[sample_idx]
print(f"Predicted Label: {predicted_label[0]}, True Label: {true_label}")
