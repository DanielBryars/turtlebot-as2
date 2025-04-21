
'''
   Author: Abdulrahman Altahhan, 2025.
   version: 3.4

    This library of functionality in RL that aims for simplicity and general insight into how algorithms work, these libraries 
    are written from scratch using standard Python libraries (numpy, matplotlib etc.).
    Please note that you will need permission from the author to use the code for research, commercially or otherwise.
'''

'''
    This library implement a nonlinear function approximation for well-known 
    RL algorithms. It works by inheriting from the classes in the 
    rl.tabular library. We added nn prefix to the MRP and MDP base classes to 
    differentiate them from their ancestor but we could have kept the same names.
    As usual we start by defining an MRP class for prediction, then MDP for control
    then make other rl algorithms inherit forn them as needed.
'''

from rl.rl import *
from env.gridnn import *
from env.mountainln import *
# ===============================================================================================
import time
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt
from numpy.random import rand
from collections import deque
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

from IPython.display import clear_output

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Zeros
# ===============================================================================================
class nnMRP(MRP):
    def __init__(self, γ=0.99, nF=512, h1=32, h2=64, nbuffer=10000, nbatch=32, rndbatch=True,
                 save_weights=1000,     # if you are saving the whole object no need to save the weights
                 load_weights=False,
                 print_=False,
                 **kw):

        super().__init__(γ=γ, **kw)
        self.nF = nF   # feature extraction is integrated within the neural network model not the env
        self.h1 = h1
        self.h2 = h2
        self.nbuffer = nbuffer
        self.nbatch = nbatch
        self.rndbatch = rndbatch

        self.print_ = print_
        self.load_weights_ = load_weights
        self.save_weights_ = save_weights  # used to save the target net every now and then

        self.update_msg = 'update %s network weights...........! %d'
        self.saving_msg = 'saving %s network weights to disk...! %d'
        self.loading_msg = 'loading %s network weights from disk...!'

    def init(self):
        self.vN = self.create_model('V')                      # create V deep network
        if self.load_weights_: self.load_weights(self.vN,'V.weights.h5') # from earlier training proces   
        self.V = self.V_

    #--------------------------------------Neural Network model related---------------------------
    ''' create a model for the V or the Q function based on net_str.
        This function creates a customisable neural network model, suitable for tasks like regression 
        or classification. It supports different input dimensions, and accomodate for cnn based 
        architecture, suitable for images, and simpler dense architecture suitable for laser beams.
        It allows for customisation of hidden layers and output units (they can be passed via the constructor).
        You can change the activation functions. The model uses the usual mean squared error loss. 
    '''
    def create_model(self, net_str):
        dim = self.env.reset().shape
        x0 = Input(dim)  # (84,84,1)
        # we use the dim of the input to specify the complexity of the model
        if len(dim) == 3:
            print('creating a cnn model')
            # assuming it is an image, hence we need a few cnn layers
            x = Conv2D(self.h1, 8, 4, activation='relu')(x0)
            x = Conv2D(self.h2, 4, 2, activation='relu')(x) if self.h2 > 0 else x
            x = Conv2D(self.h2, 3, 1, activation='relu')(x) if self.h2 > 0 else x
            x = Flatten()(x)
        else:
            print('creating a dense model- traditional nn')
            # Only one or two hidden layers
            x = Dense(self.h1, activation='relu')(x0) if self.h1 > 0 else x0
            x = Dense(self.h2, activation='relu')(x)  if self.h2 > 0 else x

        x = Dense(self.nF, activation='relu')(x)
        x = Dense(1 if net_str == 'V' else self.env.nA, activation='linear', kernel_initializer=Zeros(), bias_initializer=Zeros())(x)

        model = Model(x0, x)
        model.compile(Adam(self.α), loss='mse')
        model.summary() if net_str != 'Qn' else None
        model.net_str = net_str
        return model

    def load_weights(self, net, net_str ):
        print(self.loading_msg%net_str)
        loaded_weights = net.load_weights(net_str)
        loaded_weights.assert_consumed()

    def save_weights(self):
        print(self.saving_msg%('V ',self.t_))
        self.vN.save_weights('V.weights.h5')

    #------------------------------------- value related 易-----------------------------------
    def V_(self, s, Vs=None):
        
        # update the V network if Vs is passed
        if Vs is not None: self.vN.fit(s, Vs, verbose=False); return None
        
        # predict for one state for εgreedy, or predict for a batch of states, copy to avoid auto-grad issues
        return self.vN.predict(np.expand_dims(s, 0), verbose=0)[0] if len(s.shape)!=4 else np.copy(self.vN.predict(s, verbose=0)) 
    
    #-------------------------------------------buffer related----------------------------------
    def allocate(self):
        self.buffer = deque(maxlen=self.nbuffer)

    def store_(self, s=None,a=None,rn=None,sn=None,an=None, done=None, t=0):
        self.save_weights() if (self.save_weights_ and self.t_%self.save_weights_==0) else None
        
        self.buffer.append((s, a, rn, sn, done))
    
    # deque slicing, cannot use buffer[-nbatch:] because we are using dqueue not an array
    def slice_(self, buffer, nbatch):
        return list(islice(buffer, len(buffer)-nbatch, len(buffer)))
    
    def batch(self):
        # if nbatch==nbuffer==1 then (this should give the usual qlearning without replay buffer)
        # sample nbatch tuples (each tuple has 5 items) without replacement or obtain latest nbatch from the buffer
        # zip the tuples into one tuple of 5 items and convert each item into a np array of size nbatch 
        
        samples = sample(self.buffer, self.nbatch) if self.rndbatch else self.slice_(self.buffer, self.nbatch)
        samples = [np.array(experience) for experience in zip(*samples)]
        
        # generate a set of indices handy for filtering, to be used in online()
        inds = np.arange(self.nbatch)
        
        return samples, inds

# ===============================================================================================
class nnMDP(MDP(nnMRP)):

    # update the target network every t_qNn steps
    def __init__(self, create_vN=False, **kw):
        super().__init__(**kw)
        self.create_vN = create_vN

    def init(self):
        super().init() if self.create_vN else None  # create also vN, suitable for actor-critic
        self.qN  = self.create_model('Q')           # create main policy network
        self.qNn = self.create_model('Qn')          # create target network to estimate Q(sn)

        self.load_weights(self.qN,'Q.weights.h5') if self.load_weights_ else None # from earlier training proces
        self.load_weights(self.qNn,'Q.weights.h5') if self.load_weights_ else None # from earlier training proces

        self.Q = self.Q_

    def save_weights(self):
        super().save_weights() if self.create_vN else None   # save vN weights, for actor-critic
        print(self.saving_msg%('Q', self.t_))
        self.qN.save_weights('Q.weights.h5')
    
    def set_weights(self, net):
        print(self.update_msg%(net.net_str, self.t_))
        net.set_weights(self.qN.get_weights())
        
    #------------------------------------- policies related 易-----------------------------------
    def Q_(self, s, Qs=None):
        # update the Q networks if Qs is passed
        if Qs is not None: self.qN.fit(s, Qs, verbose=0); return None

        # predict for one state for εgreedy, or predict for a batch of states, 
        # copy to avoid auto-grad issues
        return self.qN.predict(np.expand_dims(s, 0), verbose=0)[0] if len(s.shape)!=4 \
    else np.copy(self.qN.predict(s, verbose=0))

    def Qn(self, sn, update=False):
        # update the Qn networks if Qn is passed
        if update: self.set_weights(self.qNn); return None
        # no need to expand the sn shape because we use Qn(sn) with batches of states 
        # which are already correctly shaped, unlike Q_() which can be used to predict 
        # the value of one state
        return self.qNn.predict(sn, verbose=0)
    

# ===============================================================================================
class DQN(nnMDP):
    def __init__(self, α=1e-5, t_Qn=1000, **kw): 
        print('--------------------- 易  DQN is being set up 易 -----------------------')
        super().__init__(**kw)
        self.α = α
        self.store = True
        self.t_Qn = t_Qn
        
    #-------------------------------  online learning ---------------------------------
    # update the online network in every step using a batch
    def online(self, *args):
        # no updates unless the buffer has enough samples
        if len(self.buffer) < self.nbuffer:
            print('gathering experience in the buffer no update yet')
            return
        
        # sample a tuple batch: each component is a batch of items 
        # (s and a are sets of states and actions)
        (s, a, rn, sn, dones), inds = self.batch() 

        # obtain the action-values estimation from the two networks and 
        # ensure target is 0 for terminal states
        Qs = self.Q(s)
        Qn = self.Qn(sn); Qn[dones] = 0

        # now dictate what the target should have been as per the Q-learning update rule
        Qs[inds, a] = self.γ*Qn.max(1) + rn
        
        # then update both
        self.Q(s, Qs)
        self.Qn(sn, self.t_%self.t_Qn==0)
# ===============================================================================================

# exampl of usage
# nqlearn = DQN(env=iGrid(style='maze', reward='reward_1'), 
#             seed=10, episodes=100,
#             rndbatch=False, t_Qn=500, nbuffer=32, nbatch=32, 
#             **demoGame()).interact() 