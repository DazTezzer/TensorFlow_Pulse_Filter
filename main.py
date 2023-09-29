import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # для правильной работы TensorFlow
print(tf.config.list_physical_devices('GPU'))

pd.options.plotting.backend = "matplotlib"
data = pd.read_csv('test.txt', index_col=False, sep=' ', header=None)


def restore_signal(distorted_signal, noise):
    # Преобразуем данные в формат TensorFlow
    distorted_signal = tf.expand_dims(distorted_signal, axis=0)
    noise = tf.expand_dims(noise, axis=0)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='linear', input_shape=(distorted_signal.shape[1],)),
        tf.keras.layers.Dense(distorted_signal.shape[1], activation='linear')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(noise, distorted_signal, epochs=10000, batch_size=128)
    restored_signal = model.predict(noise)

    # Сохраняем модель
    model.save("my_model_10000")

    return restored_signal

distorted_signal = data[0]
noise = data[1]
# restored_signal = restore_signal(distorted_signal, noise)
#
# print(restored_signal)
#
# print(data)
# fig = data.plot()
# data2 = pd.DataFrame()
# data2.insert(0, 0, restored_signal[0])
# data2.insert(1, 1, data[1])
# fig3 = data2.plot()
# plt.show()


loaded_model = tf.keras.models.load_model("my_model")
noise = tf.expand_dims(noise, axis=0)
restored_signal_from_loaded_model = loaded_model.predict(noise)
print(data)
fig = data.plot()
data2 = pd.DataFrame()
data2.insert(0, 0, restored_signal_from_loaded_model[0])
data2.insert(1, 1, data[1])
fig3 = data2.plot()
plt.show()