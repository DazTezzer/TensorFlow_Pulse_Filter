import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # для правильной работы TensorFlow
print(tf.config.list_physical_devices('GPU'))

pd.options.plotting.backend = "matplotlib"
data = pd.read_csv('test.txt', index_col=False, sep=' ', header=None)


def keras_restore_signal(distorted_signal, noise):
    # Преобразуем данные в формат TensorFlow
    distorted_signal = tf.expand_dims(distorted_signal, axis=0)
    noise = tf.expand_dims(noise, axis=0)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(1, 640, 1, input_shape=(distorted_signal.shape[1], 1), padding='same')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(noise, distorted_signal, epochs=500, batch_size=640)
    restored_signal = model.predict(distorted_signal)

    return restored_signal


def tensorflow_restore_signal(distorted_signal, noise):
    errors = []
    mu = 0.00000000000000001
    M = 640  # Память фильтра -M..M
    N = len(noise)  # Длина входного вектора
    X = np.zeros((N, 2 * M + 2), dtype=np.float64)
    idx = 0
    y = np.zeros(N)
    for m in range(-M, M + 1):
        X[:, idx] = np.roll(noise, m, axis=0)
        idx = idx + 1
    X[:, -1] = 1  # позволяет использовать последний коэффициент как константу к выходу фильтра
    tX = tf.cast(X, dtype=tf.float64)
    # Желаемый сигнал на выходе фильтра
    tD = tf.cast(distorted_signal, dtype=tf.float64)
    C = np.zeros((2 * M + 2, N), dtype=np.float64)
    # Создаем переменную - матрицу ( вектор )
    tC = tf.Variable(tf.zeros([2 * M + 2, 1], dtype=tf.float64))

    @tf.function
    def train_step(tX, tC, tD):
        with tf.GradientTape() as g:
            tY = tf.matmul(tX, tC)[0, 0]
            tEps = tD - tY
            tQ = tEps * tEps
            grad = g.gradient(tQ, tC)
            tC.assign(tC - tf.math.scalar_mul(mu, grad))
        return tY

    for n in range(N):
        tY = train_step(tX[n:n + 1, :], tC, tD[n])
        y[n] = tY.numpy()
        C[:, n] = tC.numpy()[:, 0]
        error = np.square(distorted_signal[n] - y[n])
        errors.append(error)
    error = pd.DataFrame(errors)
    fig_errors, az = plt.subplots()
    error.plot(ax=az)
    az.set_title('График адаптации для ошибки')
    return y


#distorted_signal = data[0].to_numpy().flatten()
distorted_signal = data.iloc[200:7000, 0].to_numpy().flatten()
# distorted_signal -= np.mean(distorted_signal)
#noise = data[2].to_numpy().flatten()
noise = data.iloc[200:7000, 2].to_numpy().flatten()
# noise -= np.mean(noise)

data = pd.DataFrame()
data.insert(0, 0, distorted_signal)
data.insert(1, 1, noise)
fig, ax = plt.subplots()
data.plot(ax=ax)
ax.set_title('До обработки')



tensorflow_restored_signal = tensorflow_restore_signal(distorted_signal, noise)


data2 = pd.DataFrame()
data2.insert(0, 0, tensorflow_restored_signal)
data2.insert(1, 1, noise)
fig2, ay = plt.subplots()
data2.plot(ax=ay)
ay.set_title('После обработки tensorflow')


keras_restored_signal = keras_restore_signal(distorted_signal, noise)

data3 = pd.DataFrame()
data3.insert(0, 0, keras_restored_signal[0].reshape(-1))
data3.insert(1, 1, noise)
fig3, ad = plt.subplots()
data3.plot(ax=ad)
ad.set_title('После обработки keras')

plt.show()
