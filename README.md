# JSNN
Pure javascript implementation of a fully connected back propagation neural network

# About

JSNN was originally written for a code.org project, however is portable enough to be used outside of that environment.  The reason JSNN implements some thing weirdly (for example having to manually implement tanh) is due to code.org's environment.  JSNN is also very nicely commented, making it a great tool to learn neural networks.

# Usage and explanation

Simply include JSNN's source file, or add the code below the file

To use the neural network, first create the neural network object.
Creating the object takes in an array, this array is the topology of the neural network, or how many neurons each layer will have.  Take note of how many input and output neurons there are, as the neural network will return arrays of that size.

```
var nn = new NNetwork([2, 4, 1]);
```

To give the neural network data, simply feed forward the data.  Note the argument array will be the same size of the input neurons, or the first element of the array when we constructed the Neural Network.

```
nn.feed_forward([1,1]);
```

After feeding forward the data, we need get the results.  This will return an array of the size of the last layer of neurons, or the last number in the array when we constructed the network.

```
var results = nn.get_results();
```

This will respond with an array of the results the neural network got.

To then train the network to see where it went wrong, we need to backprop the expected right answer.  Note we are using another array here, the size is equal to the size of the last layer of neurons, or the last number in the array when we constructed it.

```
nn.back_prop([0]);
```
