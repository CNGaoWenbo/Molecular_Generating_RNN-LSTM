#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""

Please read!
This model takes in one-hot encoded molecules. This is essentially taking a String representation of a molecule and breaking it down into individual characters (characterized input). 
It processes a sequence of vector X, and taking as input each item x[i] in the sequence.

The outputed molecules must be determined if chemically valid or not. In the future, we can check the validity by comparing molecules to the original SMILES 
input used for training. Once we find the common physiochemical features of the data, we can calculate the common physiochemical features for the data. In 
addition, executing a Principal Component Analysis (PCA) on the features, and transform the newly generated molecules accordingly. 

Model uses Tensorflow backed with Keras

Comment out specific parts of the code depending on the use, between training, saving checkpoints, sampling, etc.

"""


# In[3]:


import sys
import numpy as np
from numpy.testing import assert_allclose
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[5]:


# Opening files, extracting data, and automatically closing them (SMILES strings are conjoined together with the "\n" metatag)
filename = '100k_rndm_zinc_drugs_clean.txt'

with open(filename) as f:
    # f = [next(filename) for x in range(10000)]
    raw_text = "\n".join(line.strip() for line in f)
'''
with open(filename) as f:
    count = 0
    raw_text = ''
    while(count<10000):
        raw_text += f.readline()
        count += 1
'''


# In[6]:


# creating mapping for each char to integer, also mapping for the \n (new line) is manually inserted into the dictionaries.
unique_chars = sorted(list(set(raw_text)))
# maps each unique character as int
char_to_int = dict((c, i) for i, c in enumerate(unique_chars))
# manually updates \n
char_to_int.update({-1 : "\n"})

# int to char dictionary
int_to_char = dict((i, c) for i, c in enumerate(unique_chars))
int_to_char.update({"\n" : -1})


# In[7]:


# how many unique characters
mapping_size = len(char_to_int)
reverse_mapping_size = len(int_to_char)
print ("Size of the character to integer dictionary is: ", mapping_size)
print ("Size of the integer to character dictionary is: ", reverse_mapping_size)


# In[8]:


assert mapping_size == reverse_mapping_size


# In[9]:


# Summarize the loaded data to provide lengths for preparing datasets
n_chars = len(raw_text)
n_vocab = len(unique_chars)

print ("Total number of characters in the file is: ", n_chars)

# Preparring datasets by matching the dataset lengths (dataX will be the SMILES strings and dataY will be individual characters in the SMILE string)
seq_length = 137
dataX = []
dataY = []


# In[ ]:


for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])


# In[ ]:


n_patterns = len(dataX)


# In[14]:


# re shape input sequence X (using numpy)to be [samples, time steps, physiochemical features], input format for recurrent models
X = np.reshape(dataX, (n_patterns, seq_length, 1))

# Normalize the integers in X by dividing by the number of unique SMILES characters (a.k.a vocabulary)
#X = X / float(n_vocab)'''MemoryError: Unable to allocate 4.62 GiB for an array with shape (4530342, 137, 1) and data type float64
'''

# One-hot encode the output variable (so that they can be used to generate new SMILES after training)
Y = np_utils.to_categorical(dataY)


# In[ ]:


"""CREATING THE LSTM MODEL"""

# Create the model (simple 2 layer LSTM)
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.25))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.25))
model.add(LSTM(512, return_sequences = True))
model.add(Dropout(0.25))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(0.25))
model.add(LSTM(128))
model.add(Dropout(0.25))
model.add(Dense(Y.shape[1], activation='softmax'))


# In[ ]:


print (model.summary())


# In[13]:


# Compile the model
model.compile(loss = 'categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])


# In[13]:


# # Define checkpoints (used to save the weights at each epoch, so that the model doesn't need to be retrained)
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]


# In[4]:


# # Fit the model
model.fit(X, Y, epochs = 5, batch_size = 512, callbacks = callbacks_list)


# In[ ]:


# """TO TRAIN FROM SAVED CHECKPOINT"""
# # Load weights
# model.load_weights("weights-improvement-75-1.8144.hdf5")

# # load the model
# new_model = load_model("model.h5")
# assert_allclose(model.predict(x_train),
#                 new_model.predict(x_train),
#                 1e-5)

# # fit the model
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# new_model.fit(x_train, y_train, epochs = 100, batch_size = 64, callbacks = callbacks_list)


# In[18]:


"""GENERATING NEW SMILES"""

# Load the pre-trained network weights
filename = "weights-improvement-02-2.6436.hdf5"
model.load_weights(filename)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')


# In[19]:


# Pick a random seed from the SMILES strings
start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


# In[21]:


# Generate specified number of characters in range
for i in range(137):
    x = np.reshape(pattern, (1, len(pattern), 1))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print ("\nDone.")


# In[ ]:




