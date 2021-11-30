import pandas as pd
import numpy as np


class Lzz_norm():
    def __init__(self, x, k_th = 0.025, axis = 0):
        #x should be a 2d array
        #k_th precentage is the amount of data to substract on top and bottom
        self.k = int(np.shape(x)[axis] * k_th)
        
        x2 = np.sort(x, axis)
        if axis == 0:
            self.kmax = x2[-(self.k + 1)][None, :]
            self.kmin = x2[self.k][None, :]
        elif axis == 1:
            self.kmax = x2[:, -(self.k + 1)][:, None]
            self.kmin = x2[:, self.k][:, None]

    def limi(self, x):
        x = np.select([x>=1, x>=0, x<0],[-np.e**(-x+1)+2, x, np.e**(x)-1])
        return x
    
    def unlimi(self, x):
        x = np.select([x>=1, x>=0, x<0],[1-np.log(2-x), x, np.log(x+1)])
        return x
    
    def __call__(self, x):
        x = (x - self.kmin)/(self.kmax - self.kmin + 1e-8)
        x = self.limi(x)
        return x.astype('float32')
    
    def recover(self, x):
        x = self.unlimi(x)
        x = x*(self.kmax - self.kmin + 1e-8) + self.kmin
        return x.astype('float32')


class Dataset():
    def __init__(self, file_train, label_length, label_preprocessing, data_preprocessing):
        self.label_length = label_length
        if type(file_train) == str:
            raw_train = pd.read_csv(file_train).values
        elif type(file_train) == pd.DataFrame:
            raw_train = file_train.values
        elif type(file_train) == np.ndarray:
            raw_train = file_train
        label = raw_train[:, :label_length]
        data = raw_train[:, label_length:]

        self.label_preprocessing_choices = ['Lzz_norm', 'One_hot', None]
        self.data_preprocessing_choices = ['Lzz_norm', 'self_norm', '0-1', None]

        if label_preprocessing not in self.label_preprocessing_choices:
            raise ValueError('Label Preprocessing Method Unknown.')
        if data_preprocessing not in self.data_preprocessing_choices:
            raise ValueError('Data Preprocessing Method Unknown.')

        self.label_preprocessing = label_preprocessing
        if label_preprocessing == 'Lzz_norm':
            self.zzN_label = Lzz_norm(raw_train[:, :label_length])
        elif label_preprocessing == 'One_hot' and label_length == 1:
            self.eye = np.eye(np.max(raw_train[:, :label_length]) + 1)
        
        self.data_preprocessing = data_preprocessing
        if data_preprocessing == 'Lzz_norm':
            self.zzN_data = Lzz_norm(raw_train[:, label_length:])
        

        self.label_train= self.preprocess_label(label)
        self.data_train = self.preprocess_data(data)

        self.input_length = self.data_train.shape[1]
        self.output_length = self.label_train.shape[1]
        self.training_samples = len(self.label_train)
        print('Training_samples: %d'%(self.training_samples))

    def preprocess_label(self, label):
        if self.label_preprocessing == 'Lzz_norm':
            label_processed = self.zzN_label(label)
        elif self.label_preprocessing == 'One_hot':
            label_processed = self.eye[label[:, 0]]
        elif self.label_preprocessing == None:
            label_processed = label
        return label_processed
    
    def preprocess_data(self, data):
        if self.data_preprocessing == 'Lzz_norm':
            data_processed = self.zzN_data(data)
        elif self.data_preprocessing == 'self_norm':
            data_processed = Lzz_norm(data, axis = 1)(data)
        elif self.data_preprocessing == '0-1':
            data_processed = Lzz_norm(data, k_th = 0., axis = 1)(data)
        elif self.data_preprocessing == None:
            data_processed = data
        return data_processed

    def set_test_data(self, file_test):
        if type(file_test) == str:
            raw_test = pd.read_csv(file_test).values
        elif type(file_test) == pd.DataFrame:
            raw_test = file_test.values
        elif type(file_test) == np.ndarray:
            raw_test = file_test

        label = raw_test[:, :self.label_length]
        data = raw_test[:, self.label_length:]

        self.label_test = self.preprocess_label(label)
        self.data_test = self.preprocess_data(data)

        self.testing_samples = len(self.label_test)
        print('Testing_samples: %d'%(self.testing_samples))

    def batch_train(self, batchsize = 200):
        nums = np.array([np.random.randint(0, len(self.label_train)) for _ in range(batchsize)])
        x = self.data_train[nums]
        y = self.label_train[nums]
        return x, y

    def batch_test(self, batchsize = 200):
        nums = np.array([np.random.randint(0, len(self.label_test)) for _ in range(batchsize)])
        x = self.data_test[nums]
        y = self.label_test[nums]
        return x, y

    def recover_label(self, labels):
        if self.label_preprocessing == 'Lzz_norm':
            return self.zzN_label.recover(labels)

        elif self.label_preprocessing == 'One_hot':
            return np.argmax(labels, axis = 1)[:, None]

        elif self.label_preprocessing == None:
            return labels

    def slim(self):
        try:
            del(self.label_train)
        except:
            pass
        try:
            del(self.data_train)
        except:
            pass
        try:
            del(self.label_test)
        except:
            pass
        try:
            del(self.data_test)
        except:
            pass
        print('This Dataset has been pruned.')