import numpy as np

class NeuralNetwork: 

    def __init__(self, input_size, layer_sizes, activations):
        self.weights = []
        self.biases = []
        self.activation_functions = [] 

        self.weight_updates = []
        self.bias_updates = [] 
        self.layer_activations = []

        for layer_size, activation in zip(layer_sizes, activations): 
            self.weights.append(np.random.normal(size=(input_size, layer_size)))
            self.biases.append(np.zeros(layer_size))
            self.activation_functions.append(activation())
            input_size = layer_size

        assert len(self.activation_functions) == len(self.weights)

    def forward(self, input, store_act=False): 
        for weight, bias, act in zip(self.weights, self.biases, self.activation_functions): 
            if store_act:
                self.layer_activations.append(input)
            input = act.forward(input@weight + bias) 
        return input

    def backward(self, loss_derivative, update_weights=True, lr=0.001):
        for i in range(len(self.layer_activations)-1, -1, -1):
            loss_derivative = loss_derivative * self.activation_functions[i].backward()
            self.weight_updates.insert(0, self.layer_activations[i].T@loss_derivative)
            self.bias_updates.insert(0, np.mean(loss_derivative, axis=0))
            if i > 0:
                loss_derivative = loss_derivative@self.weights[i].T
        
        self.layer_activations = []
        if update_weights: 
            self.update_weights(lr)

    
    def update_weights(self, lr):
        for i in range(len(self.weight_updates)): 
            self.weights[i] += lr * self.weight_updates[i] 
            self.biases[i] += lr * self.bias_updates[i]
        self.weight_updates = []
        self.bias_updates = []
        

