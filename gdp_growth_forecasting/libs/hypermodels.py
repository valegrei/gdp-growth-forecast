from models import build_mlp, build_cnn, build_lstm, build_gru, build_seq2seq
from utils import shift_data
from kerastuner import HyperModel





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
        
        return build_lstm(hp_time_steps,self.n_features,self.n_steps_out,hp_cells,hp_ratio_aprendizaje,self.metrics)

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
        
        return build_gru(hp_time_steps,self.n_features,self.n_steps_out,hp_cells,hp_ratio_aprendizaje,self.metrics)

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

        return build_seq2seq(hp_time_steps,self.n_features,self.n_steps_out,hp_cells,hp_ratio_aprendizaje,self.metrics)

    def fit(self, hp, model,x,y, **kwargs):
        x_s,y_s = shift_data(x,y,hp.get('steps_in'))
        return model.fit(x=x_s,y=y_s,**kwargs)