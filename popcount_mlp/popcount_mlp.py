# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:05:15 2019

@author: Gaiger
"""

import numpy as np
import matplotlib.pyplot as plt

import copy

 
def ConvertToBitArray(value, num_bit):
    
    val = copy.deepcopy(value)
     
    bit_array = np.zeros((num_bit, 1))              
    i = 0
    
    while val > 0:
        if(val & 0x01):        
            bit_array[i] = 1
        val >>= 1    
        i += 1
     
    return bit_array


def Popcount(value, num_bit):
    
     val = copy.deepcopy(value)
      
     count = 0
     i = 0
     
     while val > 0:
         if(val & 0x01):
             count += 1            
         val >>= 1    
         i += 1
         
     return count


 
def BitToMaxNumber(bits):

    maxNumber = 0
    while(bits > 0):
        maxNumber |= 1
        maxNumber <<= 1        
        bits -= 1        
        
    maxNumber >>= 1
    
    return maxNumber         


class NeuralNetwork(object):
    def __init__(self, bitNumber, num_neurons):
        self.bitNumber = bitNumber
        self.num_neurons = num_neurons
        self.outputNumber = self.bitNumber + 1
        
        self.W1 = np.random.uniform(-1, 1, (num_neurons, bitNumber))        
        self.b1 = np.random.uniform(-1, 1, (num_neurons, 1))       
        
        self.W2 = np.random.uniform(-1, 1, (self.outputNumber, num_neurons))            
        self.b2 = np.random.uniform(-1, 1, (self.outputNumber, 1))
       
        
        
    def Softmax(self, val):
        exps = np.exp(val - np.max(val))
        return exps / np.sum(exps, axis=0)
    
    def ReLU(self, val):
        return (val > 0) * val
    
    def deRelu(self, z):
        return (z > 0) * 1        
   
    
    def cross_entropy(self, y) :
        #return -np.sum( y * np.log(self.out))
        for i in range(y.size):
            if (0 != y[i]):
                return -np.log(self.out[i])
        pass
            
    def Forward(self, x):
                  
        self.x = x
        
        self.z1 = np.dot( self.W1, self.x)
        self.z1 += self.b1        
        self.z2 = self.ReLU(self.z1)                  
        
        self.z3 = np.dot(self.W2, self.z2)        
        self.z3 += self.b2      
      
        self.out = self.Softmax(self.z3)        
        
        return self.out

    def Backward(self, y):
                
        loss = self.cross_entropy(y)         
        out_error = y - self.out                                                  
        self.z3_error = out_error
        
        self.z2_error = self.W2.T.dot(self.z3_error)
        
        self.z1_error = self.z2_error * self.deRelu(self.z2)                        
           
                       
        self.z3_W_delta = self.z3_error.dot(self.z2.T)       
        self.z3_b_delta = self.z3_error
        
        self.z1_W_delta = self.z1_error.dot(self.x.T)       
        self.z1_b_delta = self.z1_error
        
        lr = 2e-2                                       
        self.W2 += self.z3_W_delta * lr 
        self.W1 += self.z1_W_delta * lr
        
        self.b2 += self.z3_b_delta * lr
        self.b1 += self.z1_b_delta * lr
        
        return loss  
    
if __name__ == "__main__":
    
    max_epoch = 1000
    num_sample = 100
    num_bit = 8
    num_neurons = 64
    
    max_value = BitToMaxNumber(num_bit)
    print("bit number = %d, max value = %d"%(num_bit, max_value))
    x = np.random.randint(0, high = max_value, size = (num_sample, 1))
    nn = NeuralNetwork(num_bit, num_neurons)    
    
    loss = []
    
    for epoch in range(max_epoch):
        err_count = 0
        for i in range(np.size(x, 0)):
            
            y = Popcount(x[i], num_bit)        
            x_array = ConvertToBitArray(x[i], num_bit)
            
            #print("value = %d, popcount = %d "%( x_array[i], y))   
            out_array = nn.Forward(x_array)    
            
            jj = 0
            for j in range(out_array.size):
                if(out_array[j] > out_array[jj]):
                     jj = j                
           # print("predict bits = %d "% jj);                           
      
            #np.set_printoptions(suppress = True)
            #print("\npredicte = \n%s" % np.reshape(out_array, (bitNumber + 1, 1)) )
           
            yy = np.zeros((num_bit + 1, 1))
            yy[int(y)] = 1.0
            
            if(int(y) != jj): 
                err_count += 1
                
            loss.append( nn.Backward(yy))
            
        error_rate = 100.0*err_count/np.size(x, 0)                     
        print("epoch = %d, error rate = %3.1f%%, loss = %6.5f" \
              % (epoch, error_rate, loss[-1] ))
        
        if(abs(loss[-1] - loss[-2]) <= 1e-4 and loss[-1] <= 1e-4):
            break
        
        if(error_rate <= 0.1):
            break;
            
    loss = loss[0:-1:int(num_sample/2)]
    
    plt.plot( range(0, len(loss)), np.log10(loss))
    plt.show()
