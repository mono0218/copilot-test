import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# データをロード
df = pd.read_csv('spam.csv', usecols=[0, 1], encoding='latin-1')
df.columns = ['label', 'text']

# ラベルを数値に変換
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# データを訓練データとテストデータに分割
text_train, text_test, label_train, label_test = train_test_split(df['text'], df['label'], test_size=0.2)

vectorizer = TextVectorization(output_mode='int')
vectorizer.adapt(text_train.to_numpy())

# モデルを作成
model = tf.keras.models.Sequential([
    vectorizer,
    layers.Embedding(len(vectorizer.get_vocabulary()), 64, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# モデルをコンパイル
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

# モデルを訓練
model.fit(text_train, label_train, epochs=5)

# モデルを訓練
history = model.fit(text_train, label_train, epochs=10, validation_split=0.2)

# 損失と精度をプロット
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()

# モデルをエクスポート
model.save('my_saved_model')