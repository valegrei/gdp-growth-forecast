from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, LSTM, GRU, Dropout, RepeatVector, TimeDistributed, Input, dot, Activation, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.optimizers import Adam


def build_mlp(n_steps_in: int, n_features: int, n_steps_out: int, nodes: int,
              layers: int, dropout : float, learning_rate: float, activation=None,
              metrics = None):
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

    dropout : float
        Ratio de dropout

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
    model.add(Dropout(dropout))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_cnn(n_steps_in: int, n_features: int, n_steps_out: int, conv1_kernels: int,
              conv1_kernel_size: int, conv1_dropout : float, conv2_kernels: int,
              conv2_kernel_size: int, conv2_dropout : float, dense_nodes: int, 
              dense_layers : int, dense_dropout : float, learning_rate: float, 
              conv_activation = None, dense_activation = None, metrics = None):
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

    conv1_kernels : int
        Numero de kernels para Conv 1

    conv1_kernel_size : int
        Tamano de kernel para Conv 1

    conv1_dropout : float
        Ratio de dropout para Conv 1

    conv2_kernels : int
        Numero de kernels para Conv 2

    conv2_kernel_size : int
        Tamano de kernel para Conv 2

    conv2_dropout : float
        Ratio de dropout para Conv 2

    dense_nodes : int
        Numero de nodos por capa densa

    dense_layers : int
        Numero de capas densas
    
    dense_dropout : float
        Ratio de dropout para capas densas

    learning_rate : float
        Ratio de aprendizaje

    conv_activation : 
        Funcion de activacion para Conv

    dense_activation : 
        Funcion de activacion para Dense

    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo CNN construido
    '''
    model = Sequential()
    # Conv 1
    model.add(Conv1D(
        filters=conv1_kernels,
        kernel_size=conv1_kernel_size,
        activation=conv_activation, 
        padding = 'same',
        input_shape=(n_steps_in, n_features)))
    model.add(MaxPooling1D(pool_size=conv1_kernel_size, padding='same'))
    model.add(Dropout(conv1_dropout))
    # Conv 2
    model.add(Conv1D(
        filters=conv2_kernels,
        kernel_size=conv2_kernel_size,
        activation=conv_activation, 
        padding = 'same'))
    model.add(MaxPooling1D(pool_size=conv2_kernel_size, padding='same'))
    model.add(Dropout(conv2_dropout))
    # Dense
    model.add(Flatten())
    for i in range(dense_layers):
        model.add(Dense(dense_nodes, activation=dense_activation))
    model.add(Dropout(dense_dropout))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_lstm(n_steps_in: int, n_features: int, n_steps_out: int, lstm_cells: int,
                lstm_dropout : float, dense_layers : int, dense_nodes : int, 
                dense_dropout : float, learning_rate: float, dense_activation = None,
                metrics = None):
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

    lstm_cells : int
        Numero de celulas para capa LSTM
    
    lstm_dropout : float
        Ratio de dropout para capa LSTM

    dense_layers : int
        Numero de capas Dense

    dense_nodes : int
        Numero de nodos por capa Dense

    dense_dropout : float
        Ratio de dropout para capa Dense

    learning_rate : float
        Ratio de aprendizaje
    
    dense_activation : 
        Funcion de activacion.

    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo LSTM construido
    '''
    model = Sequential()
    model.add(LSTM(lstm_cells, input_shape=(n_steps_in, n_features)))
    model.add(Dropout(lstm_dropout))
    for i in range(dense_layers):
        model.add(Dense(dense_nodes, activation=dense_activation))
    model.add(Dropout(dense_dropout))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_gru(n_steps_in: int, n_features: int, n_steps_out: int, gru_cells: int,
                gru_dropout : float, dense_layers : int, dense_nodes : int, 
                dense_dropout : float, learning_rate: float, dense_activation = None,
                metrics = None):
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

    gru_cells : int
        Numero de celulas para capa LSTM
    
    gru_dropout : float
        Ratio de dropout para capa LSTM

    dense_layers : int
        Numero de capas Dense

    dense_nodes : int
        Numero de nodos por capa Dense

    dense_dropout : float
        Ratio de dropout para capa Dense

    learning_rate : float
        Ratio de aprendizaje
    
    dense_activation : 
        Funcion de activacion.

    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo LSTM construido
    '''
    model = Sequential()
    model.add(GRU(gru_cells, input_shape=(n_steps_in, n_features)))
    model.add(Dropout(gru_dropout))
    for i in range(dense_layers):
        model.add(Dense(dense_nodes, activation=dense_activation))
    model.add(Dropout(dense_dropout))
    model.add(Dense(n_steps_out))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse', metrics=metrics)
    return model


def build_seq2seq(n_steps_in: int, n_features: int, n_steps_out: int, cells: int,
                  encoder_dropout : float, decoder_dropout : float, learning_rate : float,
                  metrics = None):
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

    encoder_dropout : float
        Ratio de dropout para capa Encoder

    decoder_dropout : float
        Ratio de dropout para capa Decoder
    
    learning_rate : float
        Ratio de aprendizaje

    metrics : Any
        Lista de Metricas

    Resultado
    ---------
    model : Sequential
        Modelo GRU construido
    '''
    input_train = Input(shape= (n_steps_in, n_features))
    output_train = Input(shape= (n_steps_out, 1))
    # Codificador
    encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(cells, dropout = encoder_dropout,
            return_sequences = True, return_state = True)(input_train)
    decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
    # Decodificador
    decoder_stack_h = LSTM(cells, dropout = decoder_dropout, return_state = False,
            return_sequences = True)(decoder_input, initial_state = [encoder_last_h, encoder_last_c])
    # Atencion
    attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
    attention = Activation('softmax')(attention)
    # Contexto
    context = dot([attention, encoder_stack_h], axes = [2,1])

    decoder_combined_context = concatenate([context, decoder_stack_h])

    out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)

    model = Model(inputs = input_train, outputs=out)
    opt = Adam(learning_rate = learning_rate)
    model.compile(loss = 'mse', optimizer = opt, metrics = metrics)
    return model