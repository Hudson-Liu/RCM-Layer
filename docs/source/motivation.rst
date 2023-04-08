Motivation
==========
Ever since the inception of the Perceptron in the 1950's, **Densely-connected Feed-Forward layers** have remained a staple of the machine learning industry. 
Dense layers are used in a wide variety of models, from ResNet34 all the way to the Transformer architecture. 

Despite its prevalence, Dense layers suffer from a few fundamental inefficiencies. 
One such issue is the computational expense of finding the optimal quantity of hidden layers and nodes. 
Even with the help of complex tuning algorithms, it is nearly impossible to ascertain the perfect amount of hidden layers and nodes for a given model. 
There are a few general rules that researchers broadly follow when constructing groups of Dense layers, but these rules only give rough estimates. 
This whole tuning issue stems from a more fundamental problem: **the inefficiency of the layer structure**. 
Neural networks are organized into layers due to their ability to model nonlinear curves and act as universal approximators; as the number of nodes increases, any model with at least 1 hidden layer can get arbitrarily close to any curve. 
Theoretically, any other neural network structure can be modeled by a 3-layer FFNN, as long as it is given an unlimited amount of nodes. 
However, universal approximation is not inherently limited to layer-like structures. 
We have found that, in many cases, alternative neural network structures are more efficient than traditional layered Dense networks.

Alternative Architectures
-------------------------
An obvious alternative to layered networks is **layerless networks**. 
By creating a pool of nodes, then forming randomly distributed connections, a network unconstrained by any layer-like structure can be created (Fig 1). 

[Fig 1]

If any of these layerless networks were able to outperform a layered network as a universal approximator, it would be proof that the layered structure is not the optimal architecture. 

In order to test this hypothesis, **we compared the performance of both networks on a set of regression datasets**. 
We first created 6 unique datasets, each representing a 2-dimensional curve (Fig 2). 
In order to test a wide range of data, we used an assortment of transcendental, polynomial, and piecewise functions. 

[Fig 2]

The *x* and *y* axes were transformed into the feature vectors and output labels. 
Each dataset consisted of 100 data points, with each data point being taken at equidistantly spaced *x*-interval. 
The value of this *x*-interval was dependent on the chosen domain for that function, as shown in Fig 2. 

In terms of the neural network itself, a total of 12 nodes was allocated, with 1 input node, 1 output node, and 10 hidden nodes. 
ReLU activations were used throughout, as their popularity makes them serve as a good baseline. 
A unique layered model was created for each permutation of hidden layers (Fig 3). 
The number of connections "*c*" was tallied for each layered model.

[Fig 3]

A corresponding set of layerless networks was created for each layered model. 
This set consisted of every single possible combination of *c* connections among the 12 nodes, as can be seen in Fig 3.
