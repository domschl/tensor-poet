# tensor-poet: a Tensorflow char-rnn implementation

[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE)
<!--
[![alt text](image link)](web link)
-->
[![Tensor Poet](https://img.shields.io/badge/TF%202%20Google%20Colab-Tensor%20Poet-yellow.svg)](https://colab.research.google.com/github/domschl/tensor-poet/blob/master/tensor_poet.ipynb)

These are tensorflow implemention along the ideas of Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) as described in '[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)'.

## Overview

These Jupyter notebooks for Tensorflow 2.x trains multi-layer LSTMs on a library of texts and then generate
new text from the neural model. Through color-highlighting, source-references within
the text generated by the model are used to link to the original sources. This visualizes
how similar the generated and original texts are.

### Run notebook in Google Colab

* <a href="https://colab.research.google.com/github/domschl/tensor-poet/blob/master/tensor_poet.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" height="12" width="12" /> Run TF 2.x tensor_poet notebook in Google Colab</a> on GPU, on TPU.

### Some features

* Runs as local notebook on CPU, GPU or (with [tensorflow_plugin](https://developer.apple.com/metal/tensorflow-plugin/) on Apple hardware.
* `tensor_poet` uses the Tensorflow 2 API
* Generates samples periodically, including source-markup.
* Saves model training data periodically, allows restarts.
* Tensorboard support
* Support for dialog with the generative model

### Performance (anecdotal evidence only!)

***Note:*** This is *not* scientific benchmark data, just a single snapshot of tests with 4-layer LSTMs, about 7.5M parameters.

Platform | time per iteration | relative performance
-------- | ------------------ | --------------------
NVidia 1080ti | 155ms         | 7x
Google Colab Tesla T4 | 241ms | 4.5x
Mac mini M1 (first tests) | 1050ms          | 1x

### History

* 2021-06-10: Removed Tensorflow v1 code.
* 2021-06-09: Removed ml-compute specific code, apple M1 now usess tensorflow 2.5 pluggable device interface, which doesn't require custom code. Works only with TF 2.5 and higher, [Apple's Tensorflow Plugin](https://developer.apple.com/metal/tensorflow-plugin/) needs to be installed for Apple platforms.
* 2020-12-11: Apple M1 neural engine working with tensorflow_macos [0.1-alpha1](https://github.com/apple/tensorflow_macos)
* 2020-12-09: Fix broken text data URL (Gutenberg), renamed old v1 `tensor_poet` to `tensor_poet_old_tf_v1`, and `eager_poet` to `tensor_poet`, since eager-mode isn't useful for TPUs and MLCompute.
* 2020-11-25: TF 2.3 fixes (api change) for TPU training. First experiments with tensorflow_macos arm64/x86_64 :(apple_poet.py, not functional).
* 2020-03-18: TPU training on colab now works.
* 2020-02-11: TF 2.1 colab now does things with TPU. The secret was to move the embeddings layer to cpu.
Unfortunately, the result is just super-slow.
* 2019-11-20: TF 2.0 gpu nightly: No visible TPU in colab support progresses so far. keras.fit() still crashes, currently Tensorboard broken with nightly too.
TF 1 version: Make sure, tf 1.x is selected in colab.
* 2019-08-26: TPU/colab now at least initializes the TPU hardware, but Keras fit() still crashes.
* 2019-06-15: TPU tests with Tensorflow 2 beta, allocation of TPUs works, training errors out with recursion error.
* 2019-05-16: First (unfinished) test version for Tensorflow 2 alpha.
* 2019-05-16: Last tensorflow 1.x version, testet with 1.13.
* 2018-10-01: Adapted for tensorflow 1.11, support for Google Colab.
* 2018-05-13: Retested with tensorflow 1.8.
* 2018-03-02: Adapted for tensorflow 1.6, upcoming change to tf.nn.softmax_cross_entropy_with_logits_v2
* 2017-07-31: tested against tensorflow 1.3rc1: worked ok, for the first time the tf api did not change.
* 2017-05-19: adapted for tensorflow 1.2rc0: batch_size can't be given as tensor and used as scalar in tf-apis.
* 2017-04-12: adapted for tensorflow 1.1 changes: definition of multi-layer LSTMs changed

### Sample model

A sample model (8 layers of LSTMs with 256 neurons) was trained for 20h on four texts from [Project Gutenberg](http://www.gutenberg.org): [Pride and Prejudice_ by Jane Austen](http://www.gutenberg.org/ebooks/42671), [Wuthering Heights by Emily Brontë](http://www.gutenberg.org/ebooks/768), [The Voyage Out by Virginia Woolf](http://www.gutenberg.org/ebooks/144) and [Emma_by Jane Austen](http://www.gutenberg.org/ebooks/158)

Intermediate results after 20h of training on an NVIDIA GTX 980 Ti:

```bash
Epoch: 462.50, iter: 225000, cross-entropy: 0.378, accuracy: 0.88851
```

![training](doc/images/training.png)

The highlighters show passages of minimum 20 characters that are verbatim copies from one of the source texts.

## Implementation

* Based on the efficient implementation of LSTMs in Tensorflow 2.x
* A single model is used for training and text-generation, since dynamic_rnns became flexible enough for this
* Tensorflow 2.x has nice performance improvements for deeply nested LSTMs both on CPU and GPU (the code runs completely on GPU, if on is available). Even a laptop without GPU starts generating discernable text within a few minutes.
* Deeply nested LSTMs (e.g. 10 layers) are supported.
* Multiple source-text-files can be given for training. After text generation, color-highlighting is used to show, where the generated text is equal to some text within the source. Thus one can visualize, how free or how close the generated text follows the original training material.
* Support for different temperatures during text generation
* Tensorboard support

## Requirements

* Tensorflow
* Python 3
* Jupyter Notebook

## Output

Shown are the training labels (y:) and the prediction by the model (yp:)

```bash
Epoch: 0.00, iter: 0, cross-entropy: 4.085, accuracy: 0.07202
   y:  doing them neither | good nor harm: but he seeks their hate with
  yp: zziiipppppppppppppppprrrrrpp               nn
Epoch: 0.37, iter: 100, cross-entropy: 2.862, accuracy: 0.24243
   y: erused the note. | Hark you, sir: I'll have them very fairly bound
  yp: a      the ae    |  | AI  e    aan  a    aeee ahe  aeee aars   aeu
```

At the beginning of the training, the model bascially guesses spaces, 'a' and 'e'. After a few iterations, things start to improve:

```bash
Epoch: 27.54, iter: 5000, cross-entropy: 1.067, accuracy: 0.66178
   y:  like a babe. |  | BAPTISTA: | Well mayst thou woo, and happy be thy speed! | But be thou arm'd for some
  yp: htive a clce  |  | PRPTISTA: | Ihll,hay t thou tio  and wevly trethe fteacy |  | ut wy theu srt'd aor hume
```

Then, the model generates samples, and highlighting references to the original training text:

![training](doc/images/trainbeginning.png)

This improves over time.

## Parameter changes

To generate higher quality text, use the `param` dict:

```python
params = {
  "vocab_size": len(textlib.i2c),
  "neurons": 128,
  "layers": 2,
  "learning_rate": 1.e-3,
  "steps": 64,}
```

Increasing `neurons` to `512`, `layers` to `5` and `steps` to `100` will yield significant higher quality output.

You can add multiple text sources, by including additional file references in:

```python
textlib = TextLibrary([  # add additional texts, to train concurrently on multiple srcs:
             'data/tiny-shakespeare.txt',
])
```

Upon text generation, the original passages from the different sources are marked with different highlighting.

If your generated text becomes a single highlighted quote, then your network is overfitting (or plagiarizing the original). In our cause, plagiarizing can be addressed by reducing the net's capacity (fewer neurons), or by adding more text.

## References

* Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* See [torch-poet](https://github.com/domschl/torch-poet) for a similar implementation using PyTorch.
* See [rnnreader](https://github.com/domschl/syncognite/tree/master/rnnreader) for a pure C++ implementation (no Tensorflow) of the same idea.

### Tensorflow 2 sources

* <https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/text_generation.ipynb>
* <https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpu_and_keras.ipynb>
