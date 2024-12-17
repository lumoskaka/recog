from src.cnn_classifier import CNNClassifier
from src.utils import load_data, preprocess_data,stand,reshape
from tensorflow.keras.utils import to_categorical
import numpy as np


# 加载真实数据
data_path = 'data/optdigits-tra.xlsx' 
try:
    X, y = load_data(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()


# 数据预处理
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# 重塑为图像数据
X_train = reshape(X_train)
X_test = reshape(X_test)

data_path = 'data/optdigits-tes.xlsx'  
try:
    X_test_outer, y_test_outer = load_data(data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# 数据预处理
X_test_outer = stand(X_test_outer)

# 重塑为图像数据
X_test_outer = reshape(X_test_outer)

# 初始化 CNN 模型
cnn = CNNClassifier(input_shape=(8,8,1), num_classes=10)

# 训练模型
print("Training CNN model...")
cnn.fit(X_train, y_train, epochs=10, batch_size=32)

# 评估模型
print("Evaluating CNN model-written...")
accuracy = cnn.evaluate(X_test, y_test)
print(f"CNN Model Accuracy on Test Data: {accuracy:.3f}")


# 评估模型
print("Evaluating CNN model-nonwritten...")
accuracy = cnn.evaluate(X_test_outer, y_test_outer)
print(f"CNN Model Accuracy on Test Data: {accuracy:.3f}")

