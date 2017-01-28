# Generate network parameters of Inception v3 based retrained model to be used in iOS MPSCNNConvolution

This is based on the repo [Convert Inception v3 batch-normalized weights into weights and biases for MPSCNNConvolution](https://github.com/kakugawa/MetalCNNWeights), with the following changes:
1. A modified Python version of convert.py to deal with a retrained Inception v3 model, as described in TensorFlow's [How to Retrain Image] (https://www.tensorflow.org/how_tos/image_retraining/) and [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets);
2. Modified iOS code based on Apple's MetalImageRecognition sample to use the converted network parameters of a retrained model for better dog breed recognition than the original MetalImageRecognition sample, which uses the Inception v3 model's network parameters.

## Convert a retrained Inception v3 model

Run `python convert_doggy.py`, changed from the `convert.py` in the original repo by replacing in `def softmax_write(output_dir, dat_dir, sess):`
```
name = 'softmax'

# read
weights = sess.graph.get_tensor_by_name('softmax/weights:0').eval()
biases  = sess.graph.get_tensor_by_name('softmax/biases:0' ).eval()
```
with:
```
name = 'final_training_ops'

# read
weights = sess.graph.get_tensor_by_name('final_training_ops/weights/final_weights:0').eval()
biases  = sess.graph.get_tensor_by_name('final_training_ops/biases/final_biases:0' ).eval()
```

This will generate in the `output_doggy` folder two new files (`bias_final_training_ops.dat` and `weights_final_training_ops.dat`), along with all the previous layers’ weights and biases dat files.

#### *How did I get the right tensor name for the retrained model?
I used the following code to generate the original Inception v3 model and the retrained model's graphs to be visualized by TensorBoard:
```
import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

#INCEPTION_LOG_DIR = '/tmp/inception_v3_log'
INCEPTION_LOG_DIR = '/tmp/dog_retrained_log'

if not os.path.exists(INCEPTION_LOG_DIR):
    os.makedirs(INCEPTION_LOG_DIR)
with tf.Session() as sess:
    #model_filename = 'classify_image_graph_def.pb'
    model_filename = '../input_doggy/dog_retrained.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    writer = tf.train.SummaryWriter(INCEPTION_LOG_DIR, graph_def)
    writer.close()
```

Then you can run `tensorboard --logdir /tmp/inception_v3_log` or `tensorboard --logdir /tmp/dog_retrained_log` to find out in the TensorBoard's Graph section the details of the softmax layer of the original Inception v3 model or the last layer, which replaces the original softmax layer, in the retrained model.*

## Use network parameters of the retrained model in iOS
First, download Apple's [MetalImageRecognition sample](https://developer.apple.com/library/prerelease/content/samplecode/MetalImageRecognition/Introduction/Intro.html) - you may want to run the sample first to see how it does image recognition of the 1000 ImageNet classes.

Then, drag the two new parameter files generated, `bias_final_training_ops.dat` and `weights_final_training_ops.dat`, in the `output_doggy` folder to the iOS sample's Inception_v3_Network_Params' binaries folder in Xcode - you can also copy all the files in the `output_doggy` folder.

After that, at the end of the `init` functioin of `Inception3Net.swift`, replace:
```
fc0 = SlimMPSCNNFullyConnected(kernelWidth: 1,
                               kernelHeight: 1,
                               inputFeatureChannels: 2048,
                               outputFeatureChannels: 1008,
                               neuronFilter: nil,
                               device: device,
                               kernelParamsBinaryName: "softmax")
```
with:
```
fc0 = SlimMPSCNNFullyConnected(kernelWidth: 1,
                               kernelHeight: 1,
                               inputFeatureChannels: 2048,
                               outputFeatureChannels: 120,
                               neuronFilter: nil,
                               device: device,
                               kernelParamsBinaryName: "final_training_ops")
```                                                           
Because [the dog dataset from Stanford](http://vision.stanford.edu/aditya86/ImageNetDogs/) we used to retrain has 120 classes, we specify `outputFeatureChannels` as 120.

Finally, replace the `labels` value at the end of `Inceptioin3Net.swift` with the 120 classes of dog breeds:
```
"siberian husky",
"keeshond",
"airedale",
"german short haired pointer",
"dandie dinmont",
"whippet",
"entlebucher",
"french bulldog",
...
```

The final code of `Inception3Net.swift` is included in this repo.

Run the MetalImageRecognition app now on a device and you'll see the app does dog breed recognition more accurately than the original sample, which can also recognize dog breeds, among 1000 classes, but in a less accurately way, because our modified version uses a specifically dog breed retrained model.

## Dependencies

- [Python 2.7](https://www.python.org/)
- [numpy](http://www.numpy.org/)
- [Tensorflow](https://www.tensorflow.org/)

## Using TensorFlow APIs in iOS
An iOS app that uses Google's TensorFlow API, instead of the Apple's Metal Performance Shaders framework, to do image recognition based on an retrained Inception v3 model is described in my blog [What Kind of Dog Is It - Using TensorFlow on Mobile Device](http://jeffxtang.github.io/deep/learning,/tensorflow,/mobile,/ai/2016/09/23/mobile-tensorflow.html).
