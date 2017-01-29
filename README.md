# tensor-poet
This is a tensorflow-1.0 implemention along the ideas of Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) as described in '[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)'.

## Implementation
* Implementation is based on the efficient LSTMs using the Tensorflow 1.0 API (Python 3)
* The same model code can be used for training and generation, since dynamic_rnns became flexible enough for this
* Tensorflow 1.0 has nice performance improvements for deeply nested LSTMs both on CPU and GPU (the code runs completely on GPU, if on is available)
* Deeply nested LSTMs (e.g. 10 layers) are supported.
* Multiple source-text-files can be given for training. After text generation, color-highlighting is used to show, where the generated text is equal to some text within the source. Thus one can visualize, how free or how close the generated text follows the original training material.
* Support for different temperatures during text generation
* Tensorboard support

## Requirements
* Tensorflow 1.0
* Python 3
* Jupyter Notebook
