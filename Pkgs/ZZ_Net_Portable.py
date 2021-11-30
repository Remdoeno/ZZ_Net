import tensorflow as tf
import pickle

class Portable_ZZ():
    def __init__(self, model_loc, Ds_loc):
        self.ZZ = tf.keras.models.load_model(model_loc, compile = False)
        Dsfile = open(Ds_loc, 'rb')
        self.Ds = pickle.load(Dsfile)
        Dsfile.close()
    def __call__(self, data):
        return self.Ds.recover_label(self.ZZ(self.Ds.preprocess_data(data)))
