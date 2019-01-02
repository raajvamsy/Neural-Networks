import numpy as np, random
from matplotlib import pyplot as plt
import csv

class NeuralNetwork():

    def __init__(self,inputno, learning, neurons):
        random.seed(0)
        self.error = 1
        self.layers = len(neurons)+1
        self.inputno  = inputno
        self.neurons = neurons
        self.learningrate = learning
        self.weights = []
        for i in range(self.layers):
            if i==0:
                self.weights.append(np.random.random((self.inputno,neurons[i])))
            elif len(neurons)>i:
                self.weights.append(np.random.random((neurons[i-1],neurons[i])))
            else:
                 self.weights.append(np.random.random((neurons[i-1],1)))
    def relu(self, x):
        for i in range(0, len(x)):
            if np.mean(x[i]) > 0:
                pass
            else:
                x[i][0] = 0
        return x

    def relu_derv(self, x):
        for i in range(0, len(x)):
            if np.mean(x[i]) > 0:
                x[i][0] = 1
            else:
                x[i][0] = 0
        return x

    def train(self, input, output, iterations):
        flag = 0
        for i in range(iterations):
            out = self.think(input)
            error=[]
            delta = []
            for k in range(len(out)):
                if k == 0:
                    error.append(output - out[len(out)-1])
                    self.error=np.mean(error)
                    if(abs(self.error)<1e-2):
                        flag = 1
                        self.save()
                        break
                else:
                    error.append(delta[k-1].dot(self.weights[len(self.weights)-k].T))
                delta.append(error[k]*self.relu_derv(out[len(out)-1-k]))
            if flag==1:
                break
            for k in range(len(out)):
                    if k == 0:
                        self.weights[k]+=np.dot(input.T,delta[k])*self.learningrate
                    else:
                        self.weights[k]+=np.dot(out[k-1].T,delta[len(delta)-1-k])*self.learningrate


    def think(self, input):
        layer = []
        for i in range(len(self.weights)):
            if i == 0:
                layer.append(self.relu(np.dot(input, self.weights[i])))
            else:
                layer.append(self.relu(np.dot(layer[i-1], self.weights[i])))
        return layer

    def save(self):
        f = open('data.txt','w')
        f.write('Learning rate: '+str(self.learningrate)+' Error:'+str(self.error)+' Neurons: '+str(self.neurons)+'\n')
        for i in self.weights:
            f.write('\n'+str(i))
        f.close()

    def plot(self,input,output):
        plt.plot(input,'g',alpha=0.5)
        plt.plot(output,'r')
        plt.show()
