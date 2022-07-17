import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator

def split_array(a:np.array,test_prop:float):
    '''
    Divide un array en dos (train, test) segun la proporcion de test

    Parametros
    ----------
    a : un numpy.array
        Array al que se dividira.

    test_prop : flotante
        Proporcion de la particion de test.
    
    Retorna
    -------
    splitting : una lista de tamano 2 (train, test)
    
    '''
    b = np.random.choice(a,len(a),replace=False)
    return np.split(b,[int((1.0-test_prop)*len(b))])

def shift_data(x,y,n_steps_in):
    '''
    
    '''
    generator = TimeseriesGenerator(x, y, length=n_steps_in, batch_size=1)
    x_s,y_s = generator[0]
    for i in range(1,len(generator)):
        x, y = generator[i]
        x_s = np.vstack((x_s,x))
        y_s = np.vstack((y_s,y))
    return x_s,y_s