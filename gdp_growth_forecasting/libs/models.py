from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam


def build_mlp(n_steps_in: int, n_features: int, n_steps_out: int, nodes: int,
              layers: int, learning_rate: float, activation=None, metrics = None):
    '''
    Construye un Perceptron Multicapa (MLP)

    Parametros
    ----------
    n_steps_in : int
        Pasos de tiempo a pasado de entrada
    n_features : int
        Numero de caracteristicas de la entrada
    n_steps_out : int
        Pasos de tiempo a futuro de salida
    nodes : int
        Numero de nodos por capa oculta
    layers : int
        Numero de capas ocultas
    learning_rate : float
        Ratio de aprendizaje
    activation : 
        Funcion de activacion.
    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo MLP construido
    '''
    model = Sequential()
    model.add(Flatten(input_shape=(n_steps_in, n_features)))
    for i in range(layers):
        model.add(Dense(nodes, activation=activation))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_cnn(n_steps_in: int, n_features: int, n_steps_out: int, kernels: int,
              kernel_size: int, nodes: int, learning_rate: float, activation=None, metrics = None):
    '''
    Construye una Red Convolucional (CNN)

    Parametros
    ----------
    n_steps_in : int
        Pasos de tiempo a pasado de entrada
    n_features : int
        Numero de caracteristicas de la entrada
    n_steps_out : int
        Pasos de tiempo a futuro de salida
    kernels : int
        Numero de kernels
    kernel_size : int
        Tamano de kernel
    nodes : int
        Numero de nodos por capa oculta
    learning_rate : float
        Ratio de aprendizaje
    activation : 
        Funcion de activacion.
    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo CNN construido
    '''
    model = Sequential()
    model.add(Conv1D(
        filters=kernels,
        kernel_size=kernel_size,
        activation=activation, input_shape=(n_steps_in, n_features)))
    model.add(MaxPooling1D(pool_size=kernel_size, padding='same'))
    model.add(Flatten())
    model.add(Dense(nodes, activation=activation))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_lstm(n_steps_in: int, n_features: int, n_steps_out: int, cells: int,
               layers_lstm: int, learning_rate: float, activation=None, metrics = None):
    '''
    Construye una Red LSTM

    Parametros
    ----------
    n_steps_in : int
        Pasos de tiempo a pasado de entrada
    n_features : int
        Numero de caracteristicas de la entrada
    n_steps_out : int
        Pasos de tiempo a futuro de salida
    cells : int
        Numero de celulas por capa recurrente
    layers_lstm : int
        Numero de capas recurrentes
    learning_rate : float
        Ratio de aprendizaje
    activation : 
        Funcion de activacion.
    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo LSTM construido
    '''
    return_sequences = True
    model = Sequential()
    model.add(LSTM(cells, activation=activation,
              return_sequences=return_sequences, input_shape=(n_steps_in, n_features)))
    for i in range(layers_lstm-1):
        if i == layers_lstm-2:
            return_sequences = False
        model.add(LSTM(cells, return_sequences=return_sequences,
                  activation=activation))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_gru(n_steps_in: int, n_features: int, n_steps_out: int, cells: int, layers: int,
              learning_rate: float, activation=None, metrics = None):
    '''
    Construye una Red GRU

    Parametros
    ----------
    n_steps_in : int
        Pasos de tiempo a pasado de entrada
    n_features : int
        Numero de caracteristicas de la entrada
    n_steps_out : int
        Pasos de tiempo a futuro de salida
    cells : int
        Numero de celulas por capa recurrente
    layers : int
        Numero de capas recurrentes
    learning_rate : float
        Ratio de aprendizaje
    activation : 
        Funcion de activacion.
    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo GRU construido
    '''
    return_sequences = True
    model = Sequential()
    model.add(GRU(cells, activation=activation,
              return_sequences=return_sequences, input_shape=(n_steps_in, n_features)))
    for i in range(layers-1):
        if i == layers-2:
            return_sequences = False
        model.add(GRU(cells, return_sequences=return_sequences,
                  activation=activation))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_seq2seq(n_steps_in: int, n_features: int, n_steps_out: int, cells: int,
                  learning_rate: float, activation=None, metrics = None):
    '''
    Construye una Red Encoder-Decoder o Seq2Seq

    Parametros
    ----------
    n_steps_in : int
        Pasos de tiempo a pasado de entrada
    n_features : int
        Numero de caracteristicas de la entrada
    n_steps_out : int
        Pasos de tiempo a futuro de salida
    cells : int
        Numero de celulas por capa recurrente
    learning_rate : float
        Ratio de aprendizaje
    activation : 
        Funcion de activacion.
    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo GRU construido
    '''
    model = Sequential()
    # Encoder
    model.add(LSTM(cells, activation=activation,
              input_shape=(n_steps_in, n_features)))
    model.add(RepeatVector(n_steps_out))
    # Decoder
    model.add(LSTM(cells, activation=activation, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model