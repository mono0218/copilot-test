import tensorflow as tf
import np as np

# モデルをロード
model = tf.keras.models.load_model('my_saved_model')

# 推論を行う文字列
input_string = "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat..."

input_data = np.array([[input_string]])

# 推論を行う
predictions = model.predict(input_data)

print(predictions)