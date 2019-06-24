# -*- coding: utf-8 -*-
r"""
Sequence Models and Long-Short Term Memory Networks
===================================================

At this point, we have seen various feed-forward networks. That is,
there is no state maintained by the network at all. This might not be
the behavior we want. Sequence models are central to NLP: they are
models where there is some sort of dependence through time between your
inputs. The classical example of a sequence model is the Hidden Markov
Model for part-of-speech tagging. Another example is the conditional
random field

A recurrent neural network is a network that maintains some kind of
state. For example, its output could be used as part of the next input,
so that information can propogate along as the network passes over the
sequence. In the case of an LSTM, for each element in the sequence,
there is a corresponding *hidden state* :math:`h_t`, which in principle
can contain information from arbitrary points earlier in the sequence.
We can use the hidden state to predict words in a language model,
part-of-speech tags, and a myriad of other things.


LSTM's in Pytorch
~~~~~~~~~~~~~~~~~

Before getting to the example, note a few things. Pytorch's LSTM expects
all of its inputs to be 3D tensors. The semantics of the axes of these
tensors is important. The first axis is the sequence itself, the second
indexes instances in the mini-batch, and the third indexes elements of
the input. We haven't discussed mini-batching, so lets just ignore that
and assume we will always have just 1 dimension on the second axis. If
we want to run the sequence model over the sentence "The cow jumped",
our input should look like

.. math::


   \begin{bmatrix}
   \overbrace{q_\text{The}}^\text{row vector} \\
   q_\text{cow} \\
   q_\text{jumped}
   \end{bmatrix}

Except remember there is an additional 2nd dimension with size 1.

In addition, you could go through the sequence one at a time, in which
case the 1st axis will have size 1 also.

Let's see a quick example.
"""

# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
#import torchvision
#import torchvision.transforms as transforms
import csv
import numpy as np
import torchvision
from torchvision import transforms
from logger import Logger


torch.manual_seed(1)

######################################################################
'''
lstm = nn.LSTM(3, 3)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3),
          torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
'''
# Here we define our model as a class

#############Define Variables########################################
input_size=1
hidden_size=64
output_dim=1
num_layers=3
num_epochs=75
learning_rate=1
#sequence_length = 1021 #1021#number of samples the network will see?  # best = 5  # overfit = 50
batch_size = 1
test_batch_size=1#57#=seq_length
#57
####################--------------------#############################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True


###################Loading the Data###############################
################################
filename='seq_length_1000.csv'




#################################

x_train=[]
y_train=[]
with open(filename,'r') as csvfile:
    csv_reader=csv.reader(csvfile,delimiter=',')
    for row in csv_reader: 
        #print('row',len(row))
        #print('row-before',len(row))  
        
        row=[x for x in row if x!='']
        x_train.append(row[2:-1])# first 2 columns are not required- patient id and posture
        y_train.append(row[-1]) 
        #print('row-after',len(row))  
        
        '''
        since we cannot convert a variable length list to a numpy array, we keep out input as list. Hence since it is diifcult to slice over the 2nd dimension of 
        the list, we remove first 2 column before appending.
        '''
#print((x_train[0][0]))
#print((y_train[0]))
x_train=x_train[1:] #first row is headings.so avoid it. 
y_train=y_train[1:]

#print((y_train[0]))
#print((x_train[0][0]))

#y_train=x_train[:,-1]# last column is y labels.
#y_train=np.expand_dims(y_train,axis=1)#126X1
#x_train=x_train[:,:-1]# remove the last column from #126X1021

'''
There are a total of 126X1021=128,646 data points.
experiment 1: seq length=6. Thus we reshape x_train to 21441X6 
'''


#num_samples=int(((x_train.shape[0]*x_train.shape[1])/sequence_length))
#print(num_samples)
#x_train=np.reshape(x_train,(num_samples,sequence_length))

#print(len(x_train[9]))
#print(len(y_train))
#x_train=np.expand_dims(x_train,axis=2)
#print(x_train.shape)#126X1021== NXTXD
#x_train=x_train.astype(np.float64)

x_train=[[float(val) for val  in sublist ] for sublist in x_train ]
y_train=[float(val) for val  in y_train ]
y_train=np.array(y_train)
y_train=np.expand_dims(y_train,axis=1)#126X1
y_train=y_train.tolist()
#temp=np.array(x_train[0])
#print(temp.shape)
#print((y_train[0][:]))
#print(len(y_train[0]))

x_test=x_train
y_test=y_train


##############################XXXXXXXXXXXXXXX#########################





#######################Data Loaders##################################

#print(x_train.shape)
#print(y_train.shape)

#print(x_train)
class TrainDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.len_dataset = len(x_train)#126

    def __getitem__(self,index):
                #Convert to tensor
                #print(y_train.shape)
                #print(index)
                #self.x_train_tensor = torch.from_numpy(x_train[index,:]).to(device)
                #x_train=np.array(x_train[index])

                self.x_train_tensor = torch.FloatTensor(x_train[index][:]).to(device)

                #self.x_train_tensor = self.x_train_tensor.float()
                self.y_train_tensor = torch.FloatTensor(y_train[index][:]).to(device) #raw/filtered
                
                #self.y_train_tensor = torch.from_numpy(y_train[index]).to(device) #raw/filtered
                #self.y_train_tensor = self.y_train_tensor.float()
                return self.x_train_tensor, self.y_train_tensor

    def __len__(self):
        return self.len_dataset

class TestDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.len_dataset = len(x_test)# 70#1200#570

    def __getitem__(self,index):
        #Convert to tensor
                #self.x_test_tensor = torch.from_numpy(x_test[index,:]).to(device)
                
                self.x_test_tensor = torch.FloatTensor(x_test[index][:]).to(device)
                #self.x_test_tensor = self.x_test_tensor.float()
                #self.y_test_tensor = torch.from_numpy(y_test[index,:]).to(device)
                
                self.y_test_tensor = torch.FloatTensor(y_test[index][:]).to(device) #raw/filtered
                #self.y_test_tensor = self.y_test_tensor.float()
                 #print(self.y_test_tensor.shape,'y_test_tensor')
                return self.x_test_tensor, self.y_test_tensor

    def __len__(self):
        return self.len_dataset

#objects to dataset classes
train_dataset = TrainDataset()
test_dataset = TestDataset()

#Initializing the dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=test_batch_size,
                                            shuffle=False)






###########################-------------##############################
class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_size, output_size=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_size = input_size# 1
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers#2

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

        # Define the output layer   
        self.linear = nn.Linear(self.hidden_size, output_size)

    
    def forward(self, x):
        # Forward pass through LSTM layer 
        # shape of lstm_out: [seq_length, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device) #2 for bidirectional
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

        #shape of x: (seq_len, batch, input_size)
            

        lstm_out, _ = self.lstm(x,(h0,c0)) #out: tensor of shape ( seq_length, batch_size, hidden_size)
       
        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #lstm_out=lstm_out[-1,:,:]
        #lstm_out=np.expand_dims(lstm_out,axis=0)
        #print('henlo',lstm_out.shape)

        y_pred = self.linear(lstm_out[-1,:,:])#1Xbatch_sizeXhidden_size--> 1Xbatch_sizeXoutput_size[which is 1]
        #print('henlo',y_pred.shape)
        return y_pred#1




model = LSTM(input_size, hidden_size, batch_size, output_dim, num_layers)
#model.load_state_dict(torch.load('lstm_seq_length_50.pth'))
loss_fn = torch.nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


##########tensorboard################################################

logger = Logger('./logs/run8')
##################-------------###
# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

#####################
# Train model
#####################

total_step = len(train_loader) 
for epoch in range(num_epochs):
    for i, (data,labels) in enumerate(train_loader):#data will be 1021, and label=1,
        #print(epoch)
        #data = data.reshape(-1, sequence_length, input_size).to(device) #()
        
        #print(data[0])
        #print(labels.shape)
        sequence_length=data.shape[1]
        #print('data.shape',data.shape[1])
        data = data.reshape(sequence_length,-1, input_size).to(device) #()
        #labels = labels.reshape(-1, sequence_length, num_classes).to(device)#()
        labels = labels.reshape(-1, output_dim).to(device)#()
        #print(data.shape)
        #print(labels.shape)
        # 
        data=data.float()
        labels = labels.float()

        #Forward Pass
        outputs = model(data)
        #print(outputs.shape)
        #print(outputs[:])
        #print(labels[:])
        loss = loss_fn(outputs, labels[:,:])

        #Backward and   optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if (i+1) % 2000== 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.9f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = { 'loss': loss.item() }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch+1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

        # 3. Log training images (image summary)
        '''
        info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }

        for tag, images in info.items():
            logger.image_summary(tag, images, epoch+1)
        '''
#Saving the model
torch.save(model.state_dict(),'lstm_seq_length_50.pth')

## sequence leghts differ




#=========================================================
########### INERTIA NETWORK TEST ############
#=========================================================

model.load_state_dict(torch.load('lstm_seq_length_50.pth'))
with torch.no_grad():
    correct = 0
    total = 0
    total_error=0
    true_total_error = 0
    for i,(test_data, labels) in enumerate(test_loader):

        test_sequence_length=test_data.shape[1]
        test_data = test_data.reshape(test_sequence_length, -1,  input_size).to(device)
        labels = labels.reshape(-1, output_dim).to(device)
        #true_labels = true_labels.reshape(-1, test_sequence_length, num_classes).to(device)

        outputs = model(test_data)
        error=torch.sum((abs(outputs-labels)))  #mean squared error
        #error = torch.sum(torch.abs(outputs-labels))  #mean absolute error
        total_error+=error.data             
        total+=labels.size(1)
        print(error.data)
        
print(total_error, 'total_error')
print(total, 'total')
print('mean absolute error',total_error/total)
#print('Test Accuracy of the model: {} %'.format(error / total))
