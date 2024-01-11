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
    print(noise.shape)
    noise = tf.expand_dims(noise, axis=0)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(1, 41, 1, input_shape=(distorted_signal.shape[1], 1), padding='same')
    ])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(learning_rate=0.00000000001)) # mean_squared_error - Эта функция оценивает среднеквадратичную разницу между ожидаемыми и предсказанными значениями.
    history = model.fit(noise, distorted_signal, epochs=500, batch_size=64)
    restored_signal = model.predict(distorted_signal)
    return restored_signal , history


def tensorflow_restore_signal(distorted_signal, noise):
    errors = []
    coef_changes = []
    mu = 0.00000000001
    M = 120  # Память фильтра -M..M
    N = len(noise)  # Длина входного вектора
    X = np.zeros((N, 2 * M + 2), dtype=np.float64)
    idx = 0
    y = np.zeros(N)
    for m in range(-M, M + 1): # входной сигнал сдвигается на m позиций по оси 0, то есть сдвигаются на m позиций назад.
                               # Сдвинутый сигнал сохраняется в столбец idx матрицы X
        X[:, idx] = np.roll(noise, m, axis=0)
        idx = idx + 1
    X[:, -1] = 1  # позволяет использовать последний коэффициент как константу к выходу фильтра для учета постоянного смещения сигнала
    # входной сигнал
    tX = tf.cast(X, dtype=tf.float64)
    # Желаемый сигнал на выходе фильтра
    tD = tf.cast(distorted_signal, dtype=tf.float64)
    # матрица C заполненная 0
    C = np.zeros((2 * M + 2, N), dtype=np.float64)
    # Создаем переменную - матрицу ( вектор ). Коэффициенты фильтра
    tC = tf.Variable(tf.zeros([2 * M + 2, 1], dtype=tf.float64))


    @tf.function
    def train_step(tX, tC, tD):
        with tf.GradientTape() as g:
            tY = tf.matmul(tX, tC)[0, 0] # матричиное умножение tX[m:m+1, :] и tC в тензор tY (выходной сигнал), предсказание y
            tEps = tD - tY # ошибка в точке m (разница между ожидаемым выходом и предсказанным)
            tQ = tEps * tEps # функция стоимости, квадрат ошибки tEps, оценка ошибки предсказания ( Эта функция потерь представляет квадрат разницы между ожидаемым выходом фильтра и предсказанным выходом для каждой итерации )
            grad = g.gradient(tQ, tC) # градиент функции tQ по отношению к tC
            tC.assign(tC - tf.math.scalar_mul(mu, grad)) # обновление tC с помощью градиентного спуска
        return tY, tQ

    for n in range(N):
        tY, tQ = train_step(tX[n:n + 1, :], tC, tD[n])
        y[n] = tY.numpy()
        C[:, n] = tC.numpy()[:, 0]
        errors.append(tQ.numpy())
        coef_changes.append(tC.numpy()[:, 0])
    return y, errors, coef_changes

# С помощью метода МНК происходит обучение фильтра для аппроксимации входного сигнала и запись коэффициентов фильтра
# и выходного сигнала на каждом временном шаге


#distorted_signal = data[0].to_numpy().flatten()
distorted_signal = data.iloc[200:14000, 0].to_numpy().flatten()
distorted_signal -= np.mean(distorted_signal)
#noise = data[2].to_numpy().flatten()
noise = data.iloc[200:14000, 2].to_numpy().flatten()
noise -= np.mean(noise)

data = pd.DataFrame()
data.insert(0, 0, distorted_signal)
data.insert(1, 1, noise)
fig_data, ax = plt.subplots()
data.plot(ax=ax)
ax.set_title('До обработки')
ax.legend(['Сигнал + Шум', 'Шум'])


tensorflow_restored_signal = tensorflow_restore_signal(distorted_signal, noise)

tensorflow_data = pd.DataFrame()
tensorflow_data.insert(0, 0, distorted_signal)
tensorflow_data.insert(1, 1, tensorflow_restored_signal[0])
fig_tensorflow_data, ay = plt.subplots()
tensorflow_data.plot(ax=ay, color=['#1F77B4', '#800080'])
ay.set_title('После обработки tensorflow')
ay.legend(['До', 'После'])

tensorflow_lossdata = pd.DataFrame()
tensorflow_lossdata.insert(0, 0, tensorflow_restored_signal[1])
fig_tensorflow_lossdata, ay1 = plt.subplots()
tensorflow_lossdata.plot(ax=ay1, color='black')
ay1.set_title('tensorflow loss \n Квадрат разницы между ожидаемым выходом фильтра \n и предсказанным выходом для каждой итерации')
ay1.legend(['loss'])

coef_changes = np.array(tensorflow_restored_signal[2])
num_cycles = coef_changes.shape[0]
num_coefs = coef_changes.shape[1]
fig_tensorflow_c_data, ay2 = plt.subplots()

for i in range(num_coefs):
    ay2.plot(range(num_cycles), coef_changes[:, i], label='Coefficient {}'.format(i+1))
ay2.set_title('tensorflow коэффициенты ')
ay2.legend(['c'])



keras_restored_signal = keras_restore_signal(distorted_signal, noise)

keras_data = pd.DataFrame()
keras_data.insert(0, 0, distorted_signal)
keras_data.insert(1, 1, keras_restored_signal[0][0].reshape(-1))
fig_keras_data, ad = plt.subplots()
keras_data.plot(ax=ad, color=['#1F77B4', '#800080'])
ad.set_title('После обработки keras')
ad.legend(['До', 'После'])

keras_loss_data = pd.DataFrame()
keras_loss_data.insert(0, 0, keras_restored_signal[1].history['loss'])
fig_keras_loss_data, ad1 = plt.subplots()
keras_loss_data.plot(ax=ad1)
ad1.set_title('loss keras')
ad1.legend(['loss'])


plt.show()
