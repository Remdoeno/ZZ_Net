import tensorflow as tf
from Pkgs.ZZ_Blocks import ZZ_block
from Pkgs.ZZ_Blocks import Linear_block
from Pkgs.ZZ_Blocks import Pass

class ZZ_Net(tf.keras.Model):
    def __init__(self, NN_type, task_name, inputs, outputs, nodes, levels, stacks, augmentation, activation = tf.nn.relu, normalization = tf.keras.layers.BatchNormalization, dropout_rate = 0.1):
        super(ZZ_Net, self).__init__()
        self.task_name = task_name
        self.type = NN_type

        if self.type == 'R':
            normalization = Pass
            dropout_rate = 0.

        drop_rate = [0, 0, dropout_rate]
        self.block = ZZ_block(outputs, nodes, levels, stacks, augmentation, activation, normalization, drop_rate)

        _ = self.block(tf.zeros([1, inputs]))
        print(self.block.summary())

    def train(self, x):
        x = self.block.train(x)
        if self.type == 'C':
            x = tf.nn.softmax(x)
        return x

    def call(self, x):
        x = self.block(x)
        if self.type == 'C':
            x = tf.nn.softmax(x)
        return x
    
    def visualize(self, x):
        _ = self.block.visualize(x)
    
    def training_loss(self, x, y):
        y_ = self.train(x)
        if self.type == 'C':
            loss = -tf.reduce_mean(y * tf.math.log(tf.clip_by_value(y_, 1e-32, 1e32)))
        if self.type == 'R':
            loss = tf.math.sqrt(tf.reduce_mean((y - y_)**2))
        return loss

    def testing_loss(self, x, y, b_s):
        y_ = self.predict(x, batch_size = b_s*10)
        if self.type == 'C':
            loss = -tf.reduce_mean(y * tf.math.log(tf.clip_by_value(y_, 1e-32, 1e32)))
        if self.type == 'R':
            loss = tf.math.sqrt(tf.reduce_mean((y - y_)**2))
        return loss


class ZZ_Linear(tf.keras.Model):
    def __init__(self, NN_type, task_name, inputs, outputs, augmentation, activation = tf.nn.relu):
        super(ZZ_Linear, self).__init__()
        self.task_name = task_name
        self.type = NN_type

        self.block = Linear_block(outputs, augmentation)

        _ = self.block(tf.zeros([1, inputs]))
        print(self.block.summary())

    def train(self, x):
        x = self.block.train(x)
        if self.type == 'C':
            x = tf.nn.softmax(x)
        return x

    def call(self, x):
        x = self.block(x)
        if self.type == 'C':
            x = tf.nn.softmax(x)
        return x
    
    def visualize(self, x):
        _ = self.block.visualize(x)
    
    def training_loss(self, x, y):
        y_ = self.train(x)
        if self.type == 'C':
            loss = -tf.reduce_mean(y * tf.math.log(tf.clip_by_value(y_, 1e-32, 1e32)))
        if self.type == 'R':
            loss = tf.math.sqrt(tf.reduce_mean((y - y_)**2))
        return loss

    def testing_loss(self, x, y, b_s):
        y_ = self.predict(x, batch_size = b_s*10)
        if self.type == 'C':
            loss = -tf.reduce_mean(y * tf.math.log(tf.clip_by_value(y_, 1e-32, 1e32)))
        if self.type == 'R':
            loss = tf.math.sqrt(tf.reduce_mean((y - y_)**2))
        return loss