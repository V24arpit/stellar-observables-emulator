import numpy as np
import random


# same architecture as classical observables network is followed here.
# just a change in weight and bias.this was done to counter the problem for vanishing and exploding gradients as the network uses 6 dense layers.
# this method is called xavier initialization.
class Astro_seis(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.weight = [np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:])]
        self.bias = [np.zeros((y,1)) for y in sizes[1:]]

    def feedforward(self,a):
        for w,b in zip(self.weight[:-1],self.bias[:-1]):
            a=elu(np.dot(w,a)+b)
        z=np.dot(self.weight[-1],a)+self.bias[-1]
        return z
    
    def sgd(self,training_data,val_data ,epochs,mini_batches_size,eta):
        train_list =[]
        val_list = []
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batches_size] for k in range (0,n,mini_batches_size)]
            for mini_batch in mini_batches:
                self.update_mini_batches(mini_batch,eta)
            print(f"Epoch {j+1} MSE : Train = {self.mse(training_data):.6f} , Validation {self.mse(val_data):.6f}")
            train_list.append(self.mse(training_data))
            val_list.append(self.mse(val_data))
        return train_list,val_list

            
    def update_mini_batches(self,mini_batch,eta):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]
        for x,y in mini_batch:
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
        self.weight = [w-(eta/len(mini_batch))*nw for w,nw in zip(self.weight,nabla_w)]
        self.bias = [b-(eta/len(mini_batch))*nb for b,nb in zip(self.bias,nabla_b)]

    def backprop(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.bias]
        nabla_w = [np.zeros(w.shape) for w in self.weight]

        #forward pass 
        activation = x 
        activations = [x]
        zs = []
        for b, w in zip(self.bias[:-1],self.weight[:-1]):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = elu(z)
            activations.append(activation)
        z = np.dot(self.weight[-1],activation)+self.bias[-1]
        zs.append(z)
        activation = z
        activations.append(activation)    

        # back ward pass
        delta = self.cost_derivative(activations[-1],y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())
        for l in range (2,self.num_layers):
            z=zs[-l]
            delta = np.dot(self.weight[-l+1].transpose(),delta)*elu_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta,activations[-l-1].transpose())
        return (nabla_b,nabla_w)
    
    def cost_derivative(self,output_activation,y):
        return (output_activation-y)
    
    def prediction(self,x):
        return self.feedforward(x)
    
    def mse(self,data):
        summ = 0
        for x, y in data:
            pred = self.prediction(x)
            summ += np.sum((pred-y)**2)
        return summ/len(data)
    
    def save_model(self,filename,x_mean,x_std,y_mean,y_std):
        np.savez(filename, weight = np.array(self.weight , dtype  = object),bias = np.array(self.bias , dtype= object),
                 X_mean = x_mean,
                 X_std = x_std,
                 Y_mean = y_mean,
                 Y_std = y_std)
    
    def load_model(self,filename):
        data = np.load (filename , allow_pickle = True)
        self.weight = list(data["weight"])
        self.bias = list(data["bias"])
        self.X_mean = data["X_mean"]
        self.X_std = data["X_std"]

        self.Y_mean = data["Y_mean"]
        self.Y_std = data["Y_std"]

def elu(z):
    return np.where(z>0,z,np.exp(z)-1)

def elu_prime(z):
    return np.where(z>0,1,np.exp(z))
