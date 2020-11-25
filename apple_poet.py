import numpy as np
import os
import sys
import json
import time
import datetime
import random
import pathlib
from urllib.request import urlopen
import tensorflow as tf

try:
    from tensorflow.python.compiler.mlcompute import mlcompute as mlc
    mlc_support = True
except:
    print("Non-MLCompute-version of Tensorflow")
    mlc_support = False

if mlc_support is True:
    tf.compat.v1.disable_eager_execution()
    mlc.set_mlc_device('any')

class TextLibrary:
    def __init__(self, descriptors, text_data_cache_directory=None, max=100000000):
        self.descriptors = descriptors
        self.data = ''
        self.cache_dir=text_data_cache_directory
        self.files = []
        self.c2i = {}
        self.i2c = {}
        self.total_size=0
        index = 1
        for descriptor, author, title in descriptors:
            fd = {}
            cache_name=self.get_cache_name(author, title)
            if os.path.exists(cache_name):
                is_cached=True
            else:
                is_cached=False
            valid=False
            if descriptor[:4] == 'http' and is_cached is False:
                try:
                    print(f"Downloading {cache_name}")
                    dat = urlopen(descriptor).read().decode('utf-8')
                    if dat[0]=='\ufeff':  # Ignore BOM
                        dat=dat[1:]
                    dat=dat.replace('\r', '')  # get rid of pesky LFs 
                    self.data += dat
                    self.total_size += len(dat)
                    fd["title"] = title
                    fd["author"] = author
                    fd["data"] = dat
                    fd["index"] = index
                    index += 1
                    valid=True
                    self.files.append(fd)
                except Exception as e:
                    print(f"Can't download {descriptor}: {e}")
            else:
                fd["title"] = title
                fd["author"] = author
                try:
                    if is_cached is True:
                        print(f"Reading {cache_name} from cache")
                        f = open(cache_name)
                    else:    
                        f = open(descriptor)
                    dat = f.read(max)
                    self.data += dat
                    self.total_size += len(dat)
                    fd["data"] = dat
                    fd["index"] = index
                    index += 1
                    self.files.append(fd)
                    f.close()
                    valid=True
                except Exception as e:
                    print(f"ERROR: Cannot read: {filename}: {e}")
            if valid is True and is_cached is False and self.cache_dir is not None:
                try:
                    print(f"Caching {cache_name}")
                    f = open(cache_name, 'w')
                    f.write(dat)
                    f.close()
                except Exception as e:
                    print(f"ERROR: failed to save cache {cache_name}: {e}")
                
        ind = 0
        for c in self.data:  # sets are not deterministic
            if c not in self.c2i:
                self.c2i[c] = ind
                self.i2c[ind] = c
                ind += 1
        self.ptr = 0

    def get_cache_name(self, author, title):
        if self.cache_dir is None:
            return None
        cname=f"{author} - {title}.txt"
        cache_filepath=os.path.join(self.cache_dir , cname)
        return cache_filepath
        
    def display_colored_html(self, textlist, dark_mode=False, pre='', post=''):
        bgcolorsWht = ['#d4e6e1', '#d8daef', '#ebdef0', '#eadbd8', '#e2d7d5', '#edebd0',
                    '#ecf3cf', '#d4efdf', '#d0ece7', '#d6eaf8', '#d4e6f1', '#d6dbdf',
                    '#f6ddcc', '#fae5d3', '#fdebd0', '#e5e8e8', '#eaeded', '#A9CCE3']
        bgcolorsDrk = ['#342621','#483a2f', '#3b4e20', '#2a3b48', '#324745', '#3d3b30',
                    '#3c235f', '#443f4f', '#403c37', '#463a28', '#443621', '#364b5f',
                    '#264d4c', '#2a3553', '#3d2b40', '#354838', '#3a3d4d', '#594C23']
        if dark_mode is False:
            bgcolors=bgcolorsWht
        else:
            bgcolors=bgcolorsDrk
        out = ''
        for txt, ind in textlist:
            txt = txt.replace('\n', '<br>')
            if ind == 0:
                out += txt
            else:
                out += "<span style=\"background-color:"+bgcolors[ind % 16]+";\">" + \
                       txt + "</span>"+"<sup>[" + str(ind) + "]</sup>"
        display(HTML(pre+out+post))

    def source_highlight(self, txt, minQuoteSize=10, dark_mode=False):
        tx = txt
        out = []
        qts = []
        txsrc = [("Sources: ", 0)]
        sc = False
        noquote = ''
        while len(tx) > 0:  # search all library files for quote 'txt'
            mxQ = 0
            mxI = 0
            mxN = ''
            found = False
            for f in self.files:  # find longest quote in all texts
                p = minQuoteSize
                if p <= len(tx) and tx[:p] in f["data"]:
                    p = minQuoteSize + 1
                    while p <= len(tx) and tx[:p] in f["data"]:
                        p += 1
                    if p-1 > mxQ:
                        mxQ = p-1
                        mxI = f["index"]
                        mxN = f"{f['author']}: {f['title']}"
                        found = True
            if found:  # save longest quote for colorizing
                if len(noquote) > 0:
                    out.append((noquote, 0))
                    noquote = ''
                out.append((tx[:mxQ], mxI))
                tx = tx[mxQ:]
                if mxI not in qts:  # create a new reference, if first occurence
                    qts.append(mxI)
                    if sc:
                        txsrc.append((", ", 0))
                    sc = True
                    txsrc.append((mxN, mxI))
            else:
                noquote += tx[0]
                tx = tx[1:]
        if len(noquote) > 0:
            out.append((noquote, 0))
            noquote = ''
        self.display_colored_html(out, dark_mode=dark_mode)
        if len(qts) > 0:  # print references, if there is at least one source
            self.display_colored_html(txsrc, dark_mode=dark_mode, pre="<small><p style=\"text-align:right;\">",
                                     post="</p></small>")

    def get_slice(self, length):
        if (self.ptr + length >= len(self.data)):
            self.ptr = 0
        if self.ptr == 0:
            rst = True
        else:
            rst = False
        sl = self.data[self.ptr:self.ptr+length]
        self.ptr += length
        return sl, rst

    def decode(self, ar):
        return ''.join([self.i2c[ic] for ic in ar])

    def get_random_slice(self, length):
        p = random.randrange(0, len(self.data)-length)
        sl = self.data[p:p+length]
        return sl

    def get_slice_array(self, length):
        ar = np.array([c for c in self.get_slice(length)[0]])
        return ar

    def get_encoded_slice(self, length):
        s, rst = self.get_slice(length)
        X = [self.c2i[c] for c in s]
        return X
        
    def get_encoded_slice_array(self, length):
        return np.array(self.get_encoded_slice(length))

    def get_sample(self, length):
        s, rst = self.get_slice(length+1)
        X = [self.c2i[c] for c in s[:-1]]
        y = [self.c2i[c] for c in s[1:]]
        return (X, y, rst)

    def get_random_sample(self, length):
        s = self.get_random_slice(length+1)
        X = [self.c2i[c] for c in s[:-1]]
        y = [self.c2i[c] for c in s[1:]]
        return (X, y)

    def get_sample_batch(self, batch_size, length):
        smpX = []
        smpy = []
        for i in range(batch_size):
            Xi, yi, rst = self.get_sample(length)
            smpX.append(Xi)
            smpy.append(yi)
        return smpX, smpy, rst

    def get_random_sample_batch(self, batch_size, length):
        for i in range(batch_size):
            Xi, yi = self.get_random_sample(length)
            # smpX.append(Xi)
            # smpy.append(yi)
            if i==0:
                smpX=np.array(Xi, dtype=np.float32)
                smpy=np.array(yi, dtype=np.float32)
            else:
                smpX = np.vstack((smpX, np.array(Xi, dtype=np.float32)))
                smpy = np.vstack((smpy, np.array(yi, dtype=np.float32)))
                # smpy = np.append(smpy, np.array(yi, dtype=np.float32), axis=0)
        return np.array(smpX), np.array(smpy)
    
    def get_random_onehot_sample_batch(self, batch_size, length):
        X, y = self.get_random_sample_batch(batch_size, length)
        # xoh = one_hot(X,len(self.i2c))
        xoh = tf.keras.backend.one_hot(X, len(self.i2c))
        ykc = tf.keras.backend.constant(y)
        return xoh, ykc

libdesc = {
    "name": "Women-Writers",
    "description": "A collection of works of Woolf, Austen and Brontë",
    "lib": [
        # ('data/tiny-shakespeare.txt', 'William Shakespeare', 'Some parts'),   # local file example
        # ('http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/1/0/100/100-0.txt', 'Shakespeare', 'Collected Works'),
        ('http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/3/7/4/3/37431/37431.txt', 'Jane Austen', 'Pride and Prejudice'),
        ('http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/7/6/768/768.txt', 'Emily Brontë', 'Wuthering Heights'),         
        ('http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/1/4/144/144.txt', 'Virginia Wolf', 'Voyage out'),
        ('http://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/1/5/158/158.txt', 'Jane Austen', 'Emma')
    ]
}

root_path='.'
data_cache_path=os.path.join(root_path,f"{libdesc['name']}/Data")
if data_cache_path is not None:
    pathlib.Path(data_cache_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(data_cache_path):
        print("ERROR, the cache directory does not exist. This will fail.")
    else:
        with open(os.path.join(data_cache_path,'libdesc.json'),'w') as f:
            json.dump(libdesc,f,indent=4)

textlib = TextLibrary(libdesc["lib"], text_data_cache_directory=data_cache_path)
print(f"Total size of texts: {textlib.total_size}")

SEQUENCE_LEN = 60
BATCH_SIZE = 256
STATEFUL = True
LSTM_UNITS = 512
# EMBEDDING_DIM = 64 # 120
LSTM_LAYERS = 2
NUM_BATCHES=256  # int(textlib.total_size/BATCH_SIZE/SEQUENCE_LEN)
# print(NUM_BATCHES)

dx=[]
dy=[]
for i in range(NUM_BATCHES):
    x,y=textlib.get_random_onehot_sample_batch(BATCH_SIZE,SEQUENCE_LEN)
    dx.append(x)
    dy.append(y)

data_xy=(dx,dy)
textlib_dataset=tf.data.Dataset.from_tensor_slices(data_xy)
shuffle_buffer=10000
dataset=textlib_dataset.shuffle(shuffle_buffer)
print(f"One dataset: {dataset.take(1)}")

def build_model(vocab_size, steps, lstm_units, lstm_layers, batch_size, stateful=True):
    model = tf.keras.Sequential([
        # tf.keras.layers.Embedding(vocab_size, embedding_dim,
        #                          batch_input_shape=[batch_size, None]),
        # tf.keras.layers.Flatten(),
        *[tf.keras.layers.LSTM(lstm_units,
                            # input_shape=(timesteps, data_dim)
                            batch_input_shape=[batch_size, steps, vocab_size],
                            return_sequences=True,
                            stateful=stateful,
                            recurrent_initializer='glorot_uniform')  for _ in range(lstm_layers)],
        # *[tf.keras.layers.LSTM(lstm_units,
        #                     return_sequences=True,
        #                     stateful=stateful,
        #                     recurrent_initializer='glorot_uniform') for _ in range(lstm_layers-1)],
        tf.keras.layers.Dense(vocab_size)
        ])
    return model

model = build_model(
        vocab_size = len(textlib.i2c),
        # embedding_dim=EMBEDDING_DIM,
        steps=SEQUENCE_LEN,
        lstm_units=LSTM_UNITS,
        lstm_layers=LSTM_LAYERS,
        batch_size=BATCH_SIZE,
        stateful=STATEFUL)

model.summary()

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

opti = tf.keras.optimizers.Adam(lr=0.003, clipvalue=1.0)
# opti = tf.keras.optimizers.Adam(clipvalue=0.5)
# opti=tf.keras.optimizers.SGD(lr=0.003)

def scalar_loss(labels, logits):
    bl=loss(labels, logits)
    return tf.reduce_mean(bl)

model.compile(optimizer=opti, loss=loss, metrics=[scalar_loss])

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, update_freq='batch') # , histogram_freq=1) # update_freq='epoch',

EPOCHS=20
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback, tensorboard_callback])

