import tensorflow as tf
import matplotlib.pyplot as plt

class Pass():
    def __init__(self):
        pass
    def __call__(self, x, training = False):
        return x

class Noise():
    def __init__(self, rate):
        self.rate = rate
    def __call__(self, x):
        x *= self.rate * tf.random.normal(shape = x.shape)/3 + 1
        return x

class Mask():
    def __init__(self, rate):
        self.dropout = tf.keras.layers.Dropout(rate)
    def __call__(self, x):
        return self.dropout(x, training = True)


class Starter_block(tf.keras.Model):
    def __init__(self, nodes, augmentation, dense = True):
        super(Starter_block, self).__init__()
        
        self.aug_layers = []
        for l in augmentation:
            exec('self.aug_layers.append(' + l + ')')
        
        if dense == True:
            self.layer_dense1 = tf.keras.layers.Dense(nodes)
        else:
            self.layer_dense1 = Pass()

    def train(self, x):
        for layers in self.aug_layers:
            x = layers(x)
        x1 = self.layer_dense1(x)
        return x1

    def call(self, x):
        x1 = self.layer_dense1(x)
        return x1

    def visualize(self, x):
        x1 = self.layer_dense1(x)

        plt.figure(figsize=[7, 2])
        ax1 = plt.subplot(121)
        ax1.set_title('Starter_Layer: Input')
        ax1.hist(tf.reshape(x, shape=[-1, 1]).numpy(), bins=30)
        ax2 = plt.subplot(122)
        ax2.set_title('Starter_Layer: Dense')
        ax2.hist(tf.reshape(x1, shape=[-1, 1]).numpy(), bins=30)
        plt.show()
        return x1
        
class Trans_block(tf.keras.Model):
    def __init__(self, nodes, droprate, activation, normalization):
        super(Trans_block, self).__init__()
        self.drop_layer = tf.keras.layers.Dropout(droprate)
        self.layer_dense1 = tf.keras.layers.Dense(nodes)
        self.layer_activation1 = activation
        self.layer_normalization1 = normalization()

    def train(self, x):
        x = self.drop_layer(x, training = True)
        x1 = self.layer_normalization1(x, training = True)
        x2 = self.layer_activation1(x1)
        x3 = self.layer_dense1(x2)
        x4 = tf.tile(x, [1,2])
        x5 = x3 + x4
        return x5

    def call(self, x):
        x = self.drop_layer(x, training = False)
        x1 = self.layer_normalization1(x)
        x2 = self.layer_activation1(x1)
        x3 = self.layer_dense1(x2)
        x4 = tf.tile(x, [1,2])
        x5 = x3 + x4
        return x5

    def visualize(self, x):
        x = self.drop_layer(x, training = False)
        x1 = self.layer_normalization1(x)
        x2 = self.layer_activation1(x1)
        x3 = self.layer_dense1(x2)
        x4 = tf.tile(x, [1,2])
        x5 = x3 + x4

        plt.figure(figsize=[15, 2])
        ax1 = plt.subplot(141)
        ax1.set_title('Trans_Layer: Norm')
        ax1.hist(tf.reshape(x1, shape=[-1, 1]).numpy(), bins=30)
        ax2 = plt.subplot(142)
        ax2.set_title('Trans_Layer: Activate')
        ax2.hist(tf.reshape(x2, shape=[-1, 1]).numpy(), bins=30)
        ax3 = plt.subplot(143)
        ax3.set_title('Trans_Layer: Dense')
        ax3.hist(tf.reshape(x3, shape=[-1, 1]).numpy(), bins=30)
        ax4 = plt.subplot(144)
        ax4.set_title('Trans_Layer: Add')
        ax4.hist(tf.reshape(x5, shape=[-1, 1]).numpy(), bins=30)
        plt.show()
        return x5

class Res_block(tf.keras.Model):
    def __init__(self, nodes, droprate, activation, normalization):
        super(Res_block, self).__init__()
        self.drop_layer = tf.keras.layers.Dropout(droprate)
        self.layer_dense1 = tf.keras.layers.Dense(nodes)
        self.layer_dense2 = tf.keras.layers.Dense(nodes)
        self.layer_activation1 = activation
        self.layer_activation2 = activation
        self.layer_normalization1 = normalization()
        self.layer_normalization2 = normalization()

    def train(self, x):
        x = self.drop_layer(x, training = True)
        x1 = self.layer_normalization1(x, training = True)
        x2 = self.layer_activation1(x1)
        x3 = self.layer_dense1(x2)
        x4 = self.layer_normalization2(x3, training = True)
        x5 = self.layer_activation2(x4)
        x6 = self.layer_dense2(x5)
        x7 = x + x6
        return x7

    def call(self, x):
        x = self.drop_layer(x, training = False)
        x1 = self.layer_normalization1(x)
        x2 = self.layer_activation1(x1)
        x3 = self.layer_dense1(x2)
        x4 = self.layer_normalization2(x3)
        x5 = self.layer_activation2(x4)
        x6 = self.layer_dense2(x5)
        x7 = x + x6
        return x7

    def visualize(self, x):
        x = self.drop_layer(x, training = False)
        x1 = self.layer_normalization1(x)
        x2 = self.layer_activation1(x1)
        x3 = self.layer_dense1(x2)
        x4 = self.layer_normalization2(x3)
        x5 = self.layer_activation2(x4)
        x6 = self.layer_dense2(x5)
        x7 = x + x6

        plt.figure(figsize=[15, 4.5])
        ax1 = plt.subplot(241)
        ax1.set_title('ZZ_Layer: Norm')
        ax1.hist(tf.reshape(x1, shape=[-1, 1]).numpy(), bins=30)
        ax2 = plt.subplot(242)
        ax2.set_title('ZZ_Layer: Activate')
        ax2.hist(tf.reshape(x2, shape=[-1, 1]).numpy(), bins=30)
        ax3 = plt.subplot(243)
        ax3.set_title('ZZ_Layer: Dense')
        ax3.hist(tf.reshape(x3, shape=[-1, 1]).numpy(), bins=30)
        ax4 = plt.subplot(244)
        ax5 = plt.subplot(245)
        ax5.set_title('ZZ_Layer: Norm')
        ax5.hist(tf.reshape(x4, shape=[-1, 1]).numpy(), bins=30)
        ax6 = plt.subplot(246)
        ax6.set_title('ZZ_Layer: Activate')
        ax6.hist(tf.reshape(x5, shape=[-1, 1]).numpy(), bins=30)
        ax7 = plt.subplot(247)
        ax7.set_title('ZZ_Layer: Dense')
        ax7.hist(tf.reshape(x6, shape=[-1, 1]).numpy(), bins=30)
        ax8 = plt.subplot(248)
        ax8.set_title('ZZ_Layer: Add')
        ax8.hist(tf.reshape(x7, shape=[-1, 1]).numpy(), bins=30)
        plt.show()
        return x7

class Final_block(tf.keras.Model):
    def __init__(self, nodes, droprate):
        super(Final_block, self).__init__()
        self.drop_layer = tf.keras.layers.Dropout(droprate)
        self.layer_dense1 = tf.keras.layers.Dense(nodes)

    def train(self, x):
        x = self.drop_layer(x, training = True)
        x1 = self.layer_dense1(x)
        return x1

    def call(self, x):
        x = self.drop_layer(x, training = False)
        x1 = self.layer_dense1(x)
        return x1

    def visualize(self, x):
        x = self.drop_layer(x, training = False)
        x1 = self.layer_dense1(x)

        plt.figure(figsize=[7, 2])
        ax1 = plt.subplot(121)
        ax1.set_title('Final_Layer: Dense')
        ax1.hist(tf.reshape(x1, shape=[-1, 1]).numpy(), bins=30)
        ax2 = plt.subplot(122)
        ax2.set_title('Final_Layer: Softmax')
        ax2.hist(tf.reshape(tf.nn.softmax(x1), shape=[-1, 1]).numpy(), bins=30)
        plt.show()
        return x1

class ZZ_block(tf.keras.Model):
    def __init__(self, outputs, nodes, levels, stacks, augmentation = [], activation = tf.nn.relu, normalization = Pass, drop_rate = [0.0, 0.0, 0.1]):
        super(ZZ_block, self).__init__()
        self.blocks = []
        self.blocks.append(Starter_block(nodes, augmentation))
        for i in range(0, levels):
            if i != 0:
                self.blocks.append(Trans_block(nodes*(2**i), drop_rate[0], activation, normalization))
            for j in range(stacks):
                self.blocks.append(Res_block(nodes*(2**i), drop_rate[1], activation, normalization))
        self.blocks.append(Final_block(outputs, drop_rate[2]))

    def train(self, x):
        for block in self.blocks:
            x = block.train(x)
        return x

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def visualize(self, x):
        for block in self.blocks:
            x = block.visualize(x)
        return x


class Linear_block(tf.keras.Model):
    def __init__(self, outputs, augmentation = Pass):
        super(Linear_block, self).__init__()
        self.blocks = []
        self.blocks.append(Starter_block(980908, augmentation, False))
        self.blocks.append(Final_block(outputs, 0.))

    def train(self, x):
        for block in self.blocks:
            x = block.train(x)
        return x

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def visualize(self, x):
        for block in self.blocks:
            x = block.visualize(x)
        return x