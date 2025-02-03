import tensorflow as tf

# 检查是否有 GPU 可用
if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available")
