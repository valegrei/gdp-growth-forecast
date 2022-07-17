import models
from utils import shift_data
from kerastuner import HyperModel

class MLPHyperModel(HyperModel):

    def __init__(self, n_features_in, n_steps_out, metrics = ['mae','mse','mape'], name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        self.n_features = n_features_in
        self.n_steps_out = n_steps_out
        self.metrics = metrics

    def build(self, hp):
        # Parametrizamos nro de capas, nro de nodos y ratio de aprendizaje
        hp_time_steps = hp.Int('steps_in',4,20,step=2)
        hp_capas = hp.Int('nro_capas',1,5)
        hp_nodes = hp.Int('nro_nodos',32,160,step=32)
        hp_ratio_aprendizaje = hp.Choice('ratio_aprendizaje', values=[1e-2, 1e-3, 1e-4])

        return models.build_mlp(hp_time_steps,self.n_features,self.n_steps_out,hp_nodes,hp_capas,hp_ratio_aprendizaje,self.metrics)

    def fit(self, hp, model,x,y, **kwargs):
        x_s,y_s = shift_data(x,y,hp.get('steps_in'))
        return model.fit(x=x_s,y=y_s,**kwargs)

class CNNHyperModel(HyperModel):

    def __init__(self, n_features, n_steps_out, metrics = ['mae','mse','mape'], name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.metrics = metrics

    def build(self, hp):
        # Parametrizamos tamano de kernel, nro de kernels, nro de nodos en capa MLP, y ratio de aprendizaje
        hp_time_steps = hp.Int('steps_in',4,20,step=2)
        tam_kernel = hp.Int('tam_kernel',2,3,step=1)
        nro_kernels = hp.Int('nro_kernels',32,160,step=32)
        nro_nodos = hp.Int('nro_nodos',32,160,step=32)
        hp_ratio_aprendizaje = hp.Choice('ratio_aprendizaje', values=[1e-2, 1e-3, 1e-4])
        
        return models.build_cnn(hp_time_steps,self.n_features,self.n_steps_out,nro_kernels,tam_kernel,nro_nodos,hp_ratio_aprendizaje,self.metrics)

    def fit(self, hp, model,x,y, **kwargs):
        x_s,y_s = shift_data(x,y,hp.get('steps_in'))
        return model.fit(x=x_s,y=y_s,**kwargs)

class LSTMHyperModel(HyperModel):

    def __init__(self, n_features, n_steps_out, metrics = ['mae','mse','mape'], name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.metrics = metrics

    def build(self, hp):
        #Parametrizamos nro de celulas y ratio de aprendizaje
        hp_time_steps = hp.Int('steps_in',4,20,step=2)
        hp_cells = hp.Int('nro_celulas',32,160,step=32)
        hp_ratio_aprendizaje = hp.Choice('ratio_aprendizaje', values=[1e-2, 1e-3, 1e-4])
        
        return models.build_lstm(hp_time_steps,self.n_features,self.n_steps_out,hp_cells,hp_ratio_aprendizaje,self.metrics)

    def fit(self, hp, model,x,y, **kwargs):
        x_s,y_s = shift_data(x,y,hp.get('steps_in'))
        return model.fit(x=x_s,y=y_s,**kwargs)

class GRUHyperModel(HyperModel):

    def __init__(self, n_features, n_steps_out, metrics = ['mae','mse','mape'], name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.metrics = metrics

    def build(self, hp):
        #Parametrizamos nro de celulas y ratio de aprendizaje
        hp_time_steps = hp.Int('steps_in',4,20,step=2)
        hp_cells = hp.Int('nro_celulas',32,160,step=32)
        hp_ratio_aprendizaje = hp.Choice('ratio_aprendizaje', values=[1e-2, 1e-3, 1e-4])
        
        return models.build_gru(hp_time_steps,self.n_features,self.n_steps_out,hp_cells,hp_ratio_aprendizaje,self.metrics)

    def fit(self, hp, model,x,y, **kwargs):
        x_s,y_s = shift_data(x,y,hp.get('steps_in'))
        return model.fit(x=x_s,y=y_s,**kwargs)

class Seq2SeqHyperModel(HyperModel):

    def __init__(self, n_features, n_steps_out, metrics = ['mae','mse','mape'], name=None, tunable=True):
        super().__init__(name=name, tunable=tunable)
        self.n_features = n_features
        self.n_steps_out = n_steps_out
        self.metrics = metrics

    def build(self, hp):
        #Parametrizamos nro de celulas y ratio de aprendizaje
        hp_time_steps = hp.Int('steps_in',4,20,step=2)
        hp_cells = hp.Int('nro_celulas',32,160,step=32)
        hp_ratio_aprendizaje = hp.Choice('ratio_aprendizaje', values=[1e-2, 1e-3, 1e-4])

        return models.build_seq2seq(hp_time_steps,self.n_features,self.n_steps_out,hp_cells,hp_ratio_aprendizaje,self.metrics)

    def fit(self, hp, model,x,y, **kwargs):
        x_s,y_s = shift_data(x,y,hp.get('steps_in'))
        return model.fit(x=x_s,y=y_s,**kwargs)