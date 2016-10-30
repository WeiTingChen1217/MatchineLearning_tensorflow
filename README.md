<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>
-----------------

這裏是我學習 **TensorFlow** 的地方。

#### *進入 tensorflow 環境*
```shell
$ source activate tensorflow
```
測試 tensorflow 是否正確安裝
```python
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
>>> sess.run(hello)
Hello, TensorFlow!
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
>>>
```
-----------------

## 作業

* [HW1 運行範例 MNIST for ML Beginners](https://github.com/WeiTingChen1217/MatchineLearning_tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax_2.py)
* [HW2 修改範例 MNIST for ML Beginners (->99%)](https://github.com/WeiTingChen1217/MatchineLearning_tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist_softmax_3.py)

