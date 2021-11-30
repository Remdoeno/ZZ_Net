# ZZ_Net
Easy to use NN based on tensorflow

# Ds_Builder

## Load Module
```python
from Pkgs.Ds_Builder import Dataset
```

## Instantiation
```python
Ds = Dataset(
    file_train, label_length, label_preprocessing, data_preprocessing
)
```


This class creates a special __Dataset__ to be used for __ZZ_net__ that includes common preprocessing, batching, data storagement and reverse pre-processing.

The input data should be either a .csv file's location, a DataFrame, or an array that satisfies the following requirements:
* 2 Dimensional  
* First __n__ columns should be Labels (__n__>=1)  
* Next __m__ columns should be Data (__m__>=1)  

Users should include the length of labels (n) as well.

|__Args__||
|:----|:----|
```file_train``` |A __string__ that indicates the location of the file, a __```pandas.DataFrame```__, or a __```numpy.array```__
```label_length``` |An __integer__ indicating the length/dimension of labels
```label_preprocessing```|Pre-porcessing methods for label, can choose from ```['One_hot', 'Lzz_norm', None]```
```data_preprocessing```|Pre-porcessing methods for data, can choose from ```['0-1', 'Lzz_norm', 'self_norm', None]```

## Attributes

|__Attributes__||
|:----|:----|
```Dataset.label_length```|The length of label
```Dataset.label_preprocessing_choices```|Choices for label preprocessing, if the given is not included, will raise ```ValueError```
```Dataset.data_preprocessing_choices```|Choices for data preprocessing, if the given is not included, will raise ```ValueError```
```Dataset.label_preprocessing```|The preprocessing method chosen for labels
```Dataset.data_preprocessing```|The preprocessing method chosen for data
```Dataset.label_train```|The processed label for training set
```Dataset.data_train```|The processed data for training set
```Dataset.input_length```|The dimensions of data, inputs for neural network
```Dataset.output_length```|The dimensions of label, outputs for neural network
```Dataset.training_samples```|The total training samples in training set

The following Attributes will appear after a __testing set__ is added:

|__Attributes__||
|:----|:----|
 ```Dataset.label_test```|The processed label for testing set
 ```Dataset.data_test```|The processed data for testing set

## Methods

### ```Dataset.preprocess_label(label)```
Do preprocess to label depending on ```self.label_preprocessing```.

```label``` : a ```numpy.array```

__return__: The processed label. An ```array``` with (usually) same size of input ```label```.
***

### ```Dataset.preprocess_data(data)```
Do preprocess to data depending on ```self.data_preprocessing```.

```data``` : a ```numpy.array```

__return__: The processed data. An ```array``` with same size of input ```data```.
***

### ```Dataset.set_test_data(file_test)```
Set a testing set for the Dataset.

```file_test``` : A __string__ that indicates the location of the file, a __```pandas.DataFrame```__, or a __```numpy.array```__ 
***

### ```Dataset.batch_train(batchsize = 200)```
Randomly select a number of corrisponding data and label of size ```batchsize``` from ```Dataset.data_train``` and ```Dataset.label_train```.

```batchsize```: An __integer__. Use bigger ```batchsize``` when __RAM__ is bigger.

__return__: Batched ```data``` and ```label```. Both are ```array``` of ```size = (batchsize, Dataset.input_length)``` and ```(batchsize, Dataset.output_length)```
***

### ```Dataset.batch_test(batchsize = 200)```
Randomly select a number of corrisponding data and label of size ```batchsize``` from ```Dataset.data_test``` and ```Dataset.label_test```.

```batchsize```: An __integer__. Use bigger ```batchsize``` when __RAM__ is bigger.

__return__: Batched ```data``` and ```label```. Both are ```array``` of ```size = (batchsize, Dataset.input_length)``` and ```(batchsize, Dataset.output_length)``` 
***

### ```Dataset.recover_label(labels)```
The reverse operation of ```Dataset.preprocess_label()```.

```labels```: Preprocessed labels, or output of the neural network.

__return__: The reversed pre-processed data. An ```array``` with (usually) same size of input ```labels```.
***

### ```Dataset.slim()```
Delete ```Dataset.data_train```, ```Dataset.label_train```, ```Dataset.data_test```, ```Dataset.label_test``` if they exist.
***

## Save
```python
import pickle
import copy

Ds2 = copy.deepcopy(Ds)
Ds2.slim()

Dsfile = open('Ds\\Ds_task', 'wb')
pickle.dump(Ds2, Dsfile)
Dsfile.close()
```

Use ```Pickle``` to save the slimmed ```Dataset``` (So the saved dataset can still perform ```recover_label``` or ```set_test_data``` without inheriting the training data that bring privacy issues and size problems).



# ZZ_Net_Builder

## Load Module
```python
from Pkgs.ZZ_Net_Builder import ZZ_Net
```

## Instantiation
```python
ZZ = ZZ_Net(
    NN_type, task_name, inputs, outputs, 
    nodes, levels, stacks, augmentation,
    activation, normalization, dropout_rate
)
```
__ZZ_net__ is a sub-class inheriting the __```tf.keras.Model```__ class to build an easy-to-use __Residual MLP__ neural network. It's written with __Tensorflow__, and is mainly based on the __residual structure__, along with other things like __Dropout__, __Normalization__, __Activation Functions__, etc.

>__Structure for the ZZ Neural Network:__
>* m = Levels; n = Stacks; k = Nodes
>* C: Classification; R: Regression
|ZZ_Net        |Start_Block |Res_Block  |ZZ_Block(A)     |ZZ_Block(B)|Trans_Block(A)   |Trans_Block(B)|Final_Block|
|:----:        |:----:      |:----:     |:----:          |:----:     |:----:           |:----:        |:----:     |
|Start_Block\*1|Augmentation|ZZ_Block\*n|Normalization(C)|           |Normalization(C) |Tile(\*2)     |Dropout(C) |
|Res_Block\*m  |Dense(k)    |Trans_Block|Activation      |           |Activation       |              |Dense      |
|Final_Block\*1|            |           |Dense(k)        |           |Dense(k\*2\^m)   |              |Softmax(C) |
|Output        |            |           |Normalization(C)|           |Add              |Add           |Tanh(R)    |
|              |            |           |Activation      |           |                 |              |           |
|              |            |           |Dense(k)        |           |                 |              |           |
|              |            |           |Add             |Add        |                 |              |           |


|__Args__||
|:----|:----|
```NN_type``` |A __string__ either ```'C'``` (Classification) or ```'R'``` (Regression), indicating the basic tasks for this network.
```task_name``` |A __string__ to name the neural network
```inputs```|An __integer__ indicating the input size of the network, usually obtained with ```Dataset.input_length``` (for details please see __Ds_Builder__)
```outputs```|An __integer__ indicating the output size of the network, usually obtained with ```Dataset.output_length``` (for details please see __Ds_Builder__)
```nodes```|An __integer__ indicating the amount of hidden unites in each layer, bigger ```nodes``` will give you bigger network capacity. __Recommandation: 50 ~ 200.__
```levels```|An __integer__ indicating the amount of total levels in ZZ_Net, bigger ```levels``` will give you bigger network capacity. __Recommandation: 1 ~ 3.__
```stacks```|An __integer__ indicating the amount of total stacks in each ZZ_Net levels, bigger ```stacks``` will give you bigger network capacity. __Recommandation: 1 ~ 10.__
```augmentation```|A __list__ like ```['Noise(0.1)','Mask(0.1)']```. Each element of the list do some augmentation for the input data, like adding __gaussian noise (std = 0.1)__ or __mask (10% of the total input)__. The augmentation only work during training.
```activation```|A __tensorflow.nn__ function, i.e. ```tf.nn.relu```, ```tf.nn.gelu```
```normalization```|A __tensorflow.keras.layers__ class, i.e. ```tf.keras.layers.BatchNormalization```, ```tf.keras.layers.LayerNormalization```
```dropout_rate```|A __float__ indicating the dropout rate for the last layer (final block)

## Attributes

|__Attributes__||
|:----|:----|
```ZZ_Net.task_name```|A __string__, the name of the ZZ_net
```ZZ_Net.type```|A __string__, the type of task, either ```'C'``` (Classification) or ```'R'``` (Regression)
```ZZ_Net.block```|A instantiated __tf.keras.Model__.

## Methods

### ```ZZ_Net.train(x)```
Forward function for training process.

```x``` : the input data, a __tf.tensor__.

__return__: the output of ZZ_Net during training.
***

### ```ZZ_Net.call(x)```
Forward function for testing process.

```x``` : the input data, a __tf.tensor__.

__return__: the output of ZZ_Net during testing.
***

### ```ZZ_Net.visualize(x)```
Show the distribution of each layer. How the input data is processed in each layer.

```x``` : the input data, a __tf.tensor__.

***

### ```ZZ_Net.training_loss(x, y)```
Calculate the __MSE__ for ```ZZ_Net.type == 'R'``` or __Cross Entropy__ for ```ZZ_Net.type == 'C'``` using the ```y_``` generated by ```ZZ_Net.train()``` by input ```x```.

```x``` : the input data, a __tf.tensor__.
```y``` : the real label, a __tf.tensor__.

__return__: the calculated loss.
***

### ```ZZ_Net.testing_loss(x, y)```
Calculate the __MSE__ for ```ZZ_Net.type == 'R'``` or __Cross Entropy__ for ```ZZ_Net.type == 'C'``` using the ```y_``` generated by ```ZZ_Net.call()``` by input ```x```.

```x``` : the input data, a __tf.tensor__.
```y``` : the real label, a __tf.tensor__.

__return__: the calculated loss.
***

## Save
```python
tf.keras.models.save_model(ZZ_Net, 'Saved_Models\\ZZ' + ZZ.type + '_' + ZZ.task_name)
```



# ZZ_Net_Trainer

## Load Module
```python
from Pkgs.ZZ_Net_Trainer import ZZ_Trainer
```

## Instantiation
```python
zzT = ZZ_Trainer(
    Dataset, ZZ_net, total_iteration, learning_rate = 1e-3, batch_size = 200
)
```
__ZZ_Trainer__ is specially designed for training the __ZZ_Net__.


|__Args__||
|:----|:----|
```Dataset``` |An instantiated ```Dataset```.
```ZZ_net``` |An instantiated ```ZZ_Net```.
```total_iteration```|An __integer__ indicating the total iterations for training.
```learning_rate```|A __float__ specifing the learning rate of the network.
```batch_size```|An __integer__ indicating the batchsize for training.


## Attributes

|__Attributes__||
|:----|:----|
```ZZ_Train.Ds```|An instantiated ```Dataset```.
```ZZ_Train.ZZ```|An instantiated ```ZZ_Net```.
```ZZ_Train.batch_size```|An __integer__ indicating the batchsize for training.
```ZZ_Train.training_loss```|A __list__ recording the training loss of each ```ZZ_Train.record_gap``` of iterations.
```ZZ_Train.testing_loss```|A __list__ recording the testing loss of each ```ZZ_Train.record_gap``` of iterations.
```ZZ_Train.training_accuracy```|A __list__ recording the training Accuracy of each ```ZZ_Train.record_gap``` of iterations. Exist only if ```ZZ_Train.Ds.type == 'C'```.
```ZZ_Train.testing_accuracy```|A __list__ recording the testing Accuracy of each ```ZZ_Train.record_gap``` of iterations. Exist only if ```ZZ_Train.Ds.type == 'C'```.
```ZZ_Train.trained_iterations```|An __integer__ recording the total iterations the network has been trained.
```ZZ_Train.total_iteration```|An __integer__ indicating the total iterations for training.
```ZZ_Train.show_gap```|A calculated __integer__ indicating how often to show iteration results.
```ZZ_Train.record_gap```|A calculated __integer__ indicating how often to record iteration results.

## Methods

### ```ZZ_Train.train(iterations)```
Do back probagation training to ZZ_Net for a certain amount of iterations. Record the result in ```ZZ_Train.training_loss```, ```ZZ_Train.testing_loss```, ```ZZ_Train.training_accuracy```and ```ZZ_Train.testing_accuracy``` for each ```ZZ_Train.record_gap``` iterations and show the result for each ```ZZ_Train.show_gap``` iterations.

```iterations``` : An __integer__ indicating the iterations to train the __ZZ_Net__.
***

### ```ZZ_Train.Validate(examine_list)```
Show some results for the trained or halfly trained __ZZ_Net__.

```examine_list``` : a __list__ of __integers__ to specify the important nodes for training iterations.

__return__: A __DataFrame__ to show the training_loss, testing_loss, training_accuracy and testing_accuracy on those specified nodes.
