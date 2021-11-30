import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


class Training_Function():
    def __init__(self, ZZ, total_iteration, learning_rate):
        self.ZZ = ZZ       
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps = int(total_iteration/20),
            decay_rate = 0.8,
            staircase = True
            )
        
        self.op = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule,
            beta_1 = 0.9,
            beta_2 = 0.999,
            epsilon = 1e-07,
            amsgrad = False
            )
        
        self.variables = ZZ.trainable_variables

    def __call__(self, x, y):
        with tf.GradientTape() as G:
            loss = self.ZZ.training_loss(x, y)
        variables = self.variables
        grads = G.gradient(loss, variables)
        self.op.apply_gradients(grads_and_vars = zip(grads, variables))
        return loss


class ZZ_Trainer():
    def __init__(self, Ds, ZZ, total_iteration, learning_rate, batch_size):
        self.Ds = Ds
        self.ZZ = ZZ
        self.batch_size = batch_size

        self.training_loss = []; self.testing_loss = []
        if self.ZZ.type == 'C':
            self.training_accuracy = []; self.testing_accuracy = []

        self.trained_iterations = 0
        self.total_iteration = total_iteration
        self.__trainer = Training_Function(ZZ, total_iteration, learning_rate)
        self.show_gap = np.maximum(int(self.total_iteration/50), 1)
        self.record_gap = np.maximum(int(self.total_iteration/1000), 1)

    def train(self, iterations):
        for i in range(iterations):
            if i % self.record_gap == 0:
                
                x, y = self.Ds.batch_train(self.batch_size*10)
                self.training_loss.append(self.ZZ.testing_loss(x, y, self.batch_size*2).numpy())
                if self.ZZ.type == 'C':
                    self.training_accuracy.append(tf.reduce_mean(
                        tf.cast(tf.equal(tf.argmax(self.ZZ.predict(x, self.batch_size*2), axis=-1),tf.argmax(y, axis=-1)), tf.float32)).numpy())

                x, y = self.Ds.batch_test(self.batch_size*10)
                self.testing_loss.append(self.ZZ.testing_loss(x, y, self.batch_size*2).numpy())
                if self.ZZ.type == 'C':
                    self.testing_accuracy.append(tf.reduce_mean(
                        tf.cast(tf.equal(tf.argmax(self.ZZ.predict(x, self.batch_size*2), axis=-1),tf.argmax(y, axis=-1)), tf.float32)).numpy())
                
                if i % self.show_gap == 0:
                    if self.ZZ.type == 'R':
                        print(i, ', Train: ', round(self.training_loss[-1], 5), ' , Test: ', round(self.testing_loss[-1], 5))
                    if self.ZZ.type == 'C':
                        print(i, ', Train: ', round(self.training_loss[-1], 5), ' , Test: ', round(self.testing_loss[-1], 5),
                                 ', Accuracy: ', round(self.training_accuracy[-1] * 100, 1), '% , ',
                                                 round(self.testing_accuracy[-1] * 100, 1), '%.')

            x, y = self.Ds.batch_train(self.batch_size)
            _ = self.__trainer(x, y)
            self.trained_iterations += 1

        x, y = self.Ds.data_train, self.Ds.label_train
        self.training_loss.append(self.ZZ.testing_loss(x, y, self.batch_size).numpy())
        if self.ZZ.type == 'C':
            self.training_accuracy.append(tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.ZZ.predict(x, self.batch_size), axis=-1),tf.argmax(y, axis=-1)), tf.float32)).numpy())

        x, y = self.Ds.data_test, self.Ds.label_test
        self.testing_loss.append(self.ZZ.testing_loss(x, y, self.batch_size).numpy())
        if self.ZZ.type == 'C':
            self.testing_accuracy.append(tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.ZZ.predict(x, self.batch_size), axis=-1),tf.argmax(y, axis=-1)), tf.float32)).numpy())
                

    def __Val(self, TaL, TsL, TaA, TsA, number, jobdone):
        if len(self.testing_loss) >= int(number/self.record_gap + 6):
            TaL.append(np.mean(self.training_loss[int(number/self.record_gap-5):int(number/self.record_gap+6)]))
            TsL.append(np.mean(self.testing_loss[int(number/self.record_gap-5):int(number/self.record_gap+6)]))
            if self.ZZ.type == 'C':
                TaA.append(np.mean(self.training_accuracy[int(number/self.record_gap-5):int(number/self.record_gap+6)]))
                TsA.append(np.mean(self.testing_accuracy[int(number/self.record_gap-5):int(number/self.record_gap+6)]))
        else:
            jobdone = True
            if len(self.testing_loss) >= int(number/self.record_gap - 5):
                TaL.append(self.training_loss[-1])
                TsL.append(self.testing_loss[-1])
                if self.ZZ.type == 'C':
                    TaA.append(self.training_accuracy[-1])
                    TsA.append(self.testing_accuracy[-1])
        
        return TaL,TsL,TaA,TsA, jobdone

    def Validate(self, examine_list):
        TaL,TsL,TaA,TsA = [],[],[],[]
        jobdone = False
        for num in examine_list:
            if jobdone == False:
                TaL, TsL, TaA, TsA, jobdone = self.__Val(TaL, TsL, TaA, TsA, num, jobdone)

        plt.figure(figsize = [10, 4])
        ax1 = plt.subplot(121)
        ax1.scatter(np.arange(0, len(self.training_loss), 1), self.training_loss, s = 1)
        ax1.plot((np.array(examine_list)/self.record_gap)[:len(TaL)], TaL, label = 'train')
        ax1.scatter(np.arange(0, len(self.testing_loss), 1), self.testing_loss, s = 1)
        ax1.plot((np.array(examine_list)/self.record_gap)[:len(TsL)], TsL, label = 'test')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.set_title('Loss')

        if self.ZZ.type == 'C':
            ax2 = plt.subplot(122)
            ax2.scatter(np.arange(0, len(self.training_accuracy), 1), self.training_accuracy, s = 1)
            ax2.plot((np.array(examine_list)/self.record_gap)[:len(TaA)], TaA, label = 'train')
            ax2.scatter(np.arange(0, len(self.testing_accuracy), 1), self.testing_accuracy, s = 1)
            ax2.plot((np.array(examine_list)/self.record_gap)[:len(TsA)], TsA, label = 'test')
            ax2.legend()
            ax2.set_title('Accuracy')
        plt.show()

        if self.ZZ.type == 'C':
            print(confusion_matrix(tf.argmax(self.Ds.label_test, axis=-1),
                                   tf.argmax(self.ZZ.predict(self.Ds.data_test), axis=-1), 
                                   labels=None, sample_weight=None))

            return pd.DataFrame([TaL,TsL,TaA,TsA], columns = examine_list[:len(TaL)], 
                index = ['Training_loss', 'Testing_loss', 'Training_accuracy', 'Testing_accuracy'])
        
        elif self.ZZ.type == 'R':
            return pd.DataFrame([TaL,TsL], columns = examine_list[:len(TaL)], index = ['Training_loss', 'Testing_loss'])