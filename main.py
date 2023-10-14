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
        tf.keras.layers.Conv1D(1, 40, 1, input_shape=(distorted_signal.shape[1], 1), padding='same')
    ])

    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(noise, distorted_signal, epochs=500, batch_size=64)
    restored_signal = model.predict(distorted_signal)

    return restored_signal


def tensorflow_restore_signal(distorted_signal, noise):
    errors = []
    tC_values = []
    mu = 0.00000000001
    M = 200  # Память фильтра -M..M
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
            tQ = tEps * tEps # функция стоимости, квадрат ошибки tEps, оценка ошибки предсказания
            grad = g.gradient(tQ, tC) # градиент функции tQ по отношению к tC
            tC.assign(tC - tf.math.scalar_mul(mu, grad)) # обновление tC с помощью градиентного спуска
        return tY
    # print(tC.numpy()[:, 0])
    # datatc = pd.DataFrame()
    # datatc.insert(0, 0, tC.numpy()[:, 0])
    # fig, axtc = plt.subplots()
    # datatc.plot(ax=axtc)
    # ax.set_title('tc')
    # plt.show()
    for n in range(N):
        tY = train_step(tX[n:n + 1, :], tC, tD[n])
        y[n] = tY.numpy() # значение tY присваивается элементу n выходного сигнала y
        C[:, n] = tC.numpy()[:, 0] # значение tC присваивается n-ному столбцу матрицы C
        #tC_values.append(tC.numpy()[:, 0])
        #print("tC on step", n+1, ":", tC.numpy())
        error = np.square(distorted_signal[n] - y[n])
        errors.append(error)
    error = pd.DataFrame(errors)
    fig_errors, az = plt.subplots()
    error.plot(ax=az)
    az.set_title('График адаптации для ошибки')
    # tC_values = np.array(tC_values)
    # total = []
    # total = np.array(total)
    # for i in tC_values:
    #     total = np.concatenate((total, i))
    # print(len(total))
    # coef = pd.DataFrame(total)
    # fig_coef, cf = plt.subplots()
    # coef.plot(ax=cf)
    # cf.set_title('График адаптации коэффициентов')

    return y






# С помощью метода МНК происходит обучение фильтра для аппроксимации входного сигнала и запись коэффициентов фильтра
# и выходного сигнала на каждом временном шаге


distorted_signal = data[0].to_numpy().flatten()
#distorted_signal = data.iloc[200:7000, 0].to_numpy().flatten()
distorted_signal -= np.mean(distorted_signal)
noise = data[2].to_numpy().flatten()
#noise = data.iloc[200:7000, 2].to_numpy().flatten()
noise -= np.mean(noise)

data = pd.DataFrame()
data.insert(0, 0, distorted_signal)
data.insert(1, 1, noise)
fig, ax = plt.subplots()
data.plot(ax=ax)
ax.set_title('До обработки')



tensorflow_restored_signal = tensorflow_restore_signal(distorted_signal, noise)


data2 = pd.DataFrame()
data2.insert(0, 0, distorted_signal)
data2.insert(1, 1, tensorflow_restored_signal)
fig2, ay = plt.subplots()
data2.plot(ax=ay)
ay.set_title('После обработки tensorflow')


# keras_restored_signal = keras_restore_signal(distorted_signal, noise)
#
# data3 = pd.DataFrame()
# data3.insert(0, 0, keras_restored_signal[0].reshape(-1))
# data3.insert(1, 1, noise)
# fig3, ad = plt.subplots()
# data3.plot(ax=ad)
# ad.set_title('После обработки keras')

plt.show()
