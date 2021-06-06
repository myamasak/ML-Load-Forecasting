# plot inputs and outputs
from matplotlib import pyplot as plt
import seaborn as sns
import math
import numpy as np
sns.set(rc={'figure.figsize':(5,5)})

def sigmoid(x):
    return 1/(1+math.exp(-x))

def relu(x):
	return max(0.0, x)
 
def elu(x, alpha=0.15):
    if x > 0:
        return x
    else:
        return alpha*(math.exp(x)-1)
    
def leaky_relu(x, alpha=0.15):
    if x > 0:
        return x
    else:
        return alpha*x
    
def selu(x, alpha=1.6732632423543772848170429916717, _lambda=1.0507009873554804934193349852946):
    if x > 0:
        return _lambda*x
    else:
        return _lambda*(alpha*math.exp(x)-alpha)
# define a series of inputs
# inputs = [x for x in range(-3, 4)]
inputs = np.arange(-3,3)
# calculate outputs for our inputs
output = [relu(x) for x in inputs]
# line plot of raw inputs to rectified outputs
plt.figure()
plt.title('ReLU - Rectified Linear Unit')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axvline(x=0, color='k', linewidth=0.5)
plt.axhline(y=0, color='k', linewidth=0.5)

plt.plot(inputs, output)
plt.show()

# define a series of inputs
inputs = np.arange(-3,3,0.1)
# calculate outputs for our inputs
output = [elu(x) for x in inputs]
plt.figure()
plt.title('ELU - Exponential Linear Unit')
plt.xlabel('x')
plt.ylabel('ELU(x)')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axvline(x=0, color='k', linewidth=0.5)
plt.axhline(y=0, color='k', linewidth=0.5)

plt.plot(inputs, output)
plt.show()


##
# calculate outputs for our inputs
output = [leaky_relu(x) for x in inputs]
plt.figure()
plt.title('Leaky ReLU - Leaky Rectified Linear Unit')
plt.xlabel('x')
plt.ylabel('LReLU(x)')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axvline(x=0, color='k', linewidth=0.5)
plt.axhline(y=0, color='k', linewidth=0.5)

plt.plot(inputs, output)
plt.show()

##
# calculate outputs for our inputs
output = [selu(x) for x in inputs]
plt.figure()
plt.title('SELU - Scaled Exponential Linear Unit')
plt.xlabel('x')
plt.ylabel('SELU(x)')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axvline(x=0, color='k', linewidth=0.5)
plt.axhline(y=0, color='k', linewidth=0.5)

plt.plot(inputs, output)
plt.show()


##
# calculate outputs for our inputs
output = [sigmoid(x) for x in inputs]
plt.figure()
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.axvline(x=0, color='k', linewidth=0.5)
plt.axhline(y=0, color='k', linewidth=0.5)

plt.plot(inputs, output)
plt.show()