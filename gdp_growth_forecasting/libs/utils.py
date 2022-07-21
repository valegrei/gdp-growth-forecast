import numpy as np
from pandas import DataFrame, merge
import matplotlib.pyplot as plt
import contextlib

def split_df(a: DataFrame, test_prop: float):
    '''
    Divide un dataframe en tres (train, validation, test) segun la proporcion de test

    Parametros
    ----------
    a : un Dataframe
        Dataframe al que se dividira. Debe tener doble indice (iso, year)
        
    test_prop : flotante
        Proporcion de la particion de test.

    Retorna
    -------
    splitting : una lista de dataframes de tamano 3 (train, validation, test)

    '''
    iso = a.index.get_level_values('iso').unique().to_numpy()
    df_train = DataFrame()
    df_test = DataFrame()
    for i in iso:
        df_temp = a.loc[i].copy()
        df_temp['iso'] = i
        years = df_temp.index.to_numpy()
        n = len(years)
        test_year = years[-1] - n * (test_prop)
        df_train = df_train.append(df_temp.loc[ : test_year].copy())
        df_test = df_test.append(df_temp.loc[test_year : ].copy())
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    df_train = df_train.set_index(['iso','year'])
    df_test = df_test.set_index(['iso','year'])
    return df_train, df_test


def shift_data(df_x: DataFrame, df_y: DataFrame, n_steps_in: int, n_steps_out: int):
    '''
    Genera Caracteristicas pasadas y futuras desplazando por pasos de tiempos

    Parametros
    ----------
    df_x : DataFrame
        Dataframe con datos de entrada

    df_y : DataFrame
        Dataframe con datos de salida

    n_steps_in : entero
        Numero de pasos de tiempo pasados

    n_steps_out : entero
        Numero de pasos de tiempo futuros

    Retorna
    -------
    x, y : arrays
        Devuelve arrays de datos de entrada X, y datos de salida Y

    '''
    target_col = df_y.columns.to_numpy()
    features = df_x.columns.to_numpy()
    data_shifted = merge(df_x, df_y, left_index=True, right_index=True)
    x_cols, y_cols = list(), list()
    # Lag features
    for t in range(1, n_steps_in+1):
        data_shifted[features + '_t-' +
                     str(n_steps_in-t)] = data_shifted[features].shift(n_steps_in-t)
        x_cols = [*x_cols, *((features + '_t-'+str(n_steps_in-t)).tolist())]
    # Future features
    for t in range(1, n_steps_out+1):
        data_shifted[target_col+'_t+' +
                     str(t)] = data_shifted[target_col].shift(-t)
        y_cols = [*y_cols, *((target_col + '_t+'+str(t)).tolist())]

    data_shifted = data_shifted.dropna(how='any')
    x = data_shifted[x_cols].values
    x = x.reshape(len(x), n_steps_in, len(features))  # 3D
    y = data_shifted[y_cols].values  # 2D
    return x, y


def shift_join_data(df_x: DataFrame, df_y: DataFrame, indexes, n_steps_in: int, n_steps_out: int):
    '''
    Genera Caracteristicas pasadas y futuras desplazando por pasos de tiempos
    a partir de un conjunto de doble indice

    Parametros
    ----------
    df_x : DataFrame
        Dataframe con datos de entrada

    df_y : DataFrame
        Dataframe con datos de salida

    indexes : array-like
        Lista de indices a iterar.

    n_steps_in : entero
        Numero de pasos de tiempo pasados

    n_steps_out : entero
        Numero de pasos de tiempo futuros

    Retorna
    -------
    x, y : arrays
        Devuelve arrays de datos de entrada X, y datos de salida Y

    '''
    x, y = shift_data(df_x.loc[indexes[0]], df_y.loc[indexes[0]],
                      n_steps_in, n_steps_out)
    for i in range(1, len(indexes)):
        x_i, y_i = shift_data(
            df_x.loc[indexes[i]], df_y.loc[indexes[i]], n_steps_in, n_steps_out)
        x = np.concatenate((x, x_i))
        y = np.concatenate((y, y_i))
    return x, y


def mae(orig, pred):
    # Verificar si tienen mismo shape
    if(orig.shape != pred.shape):
        raise Exception('Deben tener mismo shape')
    return np.sum(np.abs(orig - pred), axis=0)/len(orig)


def rmse(orig, pred):
    # Verificar si tienen mismo shape
    if(orig.shape != pred.shape):
        raise Exception('Deben tener mismo shape')
    return np.sqrt(np.sum((orig - pred)**2, axis=0)/len(orig))


def mape(orig, pred):
    # Verificar si tienen mismo shape
    if(orig.shape != pred.shape):
        raise Exception('Deben tener mismo shape')
    return np.sum(np.abs((orig - pred)/orig), axis=0)*100/len(orig)

def plot_history(history,metric):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(history.history[metric],label='train '+metric)
    ax.plot(history.history['val_'+metric],label='validation '+metric)
    ax.legend()

def plot_pred(orig,pred):
    y_o, y_p = orig.T, pred.T
    n = len(y_o)
    fig, axes = plt.subplots(n,figsize=(5,12))
    for i in range(n):
        line_o, = axes[i].plot(y_o[i],label='orig')
        line_p, = axes[i].plot(y_p[i],label='pred')
        axes[i].legend(handles=[line_o,line_p])

def print_hp(path,tuner):
    with open(path,'a') as o:
        with contextlib.redirect_stdout(o):
            tuner.results_summary(num_trials=1)