import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 1, 20).reshape(-1, 1)  
y_target = ((1 + 0.6 * np.sin(2 * np.pi * X / 0.7)) + 0.3 * np.sin(2 * np.pi * X)) / 2 

input_size = 1     
hidden_size = 6     
output_size = 1     
learning_rate = 0.15

W1 = np.random.randn(input_size, hidden_size)  
b1 = np.zeros((1, hidden_size))                
W2 = np.random.randn(hidden_size, output_size) 
b2 = np.zeros((1, output_size))                


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def linear(x):
    return x

def linear_derivative(x):
    return 1

epochs = 8000
for epoch in range(epochs):
 
    hidden_input = X.dot(W1) + b1
    hidden_output = tanh(hidden_input)  

  
    final_input = hidden_output.dot(W2) + b2
    y_pred = linear(final_input)        

    loss = np.mean((y_pred - y_target)**2)

    output_error = (y_pred - y_target) * linear_derivative(final_input)
    W2_gradient = hidden_output.T.dot(output_error) / X.shape[0]
    b2_gradient = np.mean(output_error, axis=0)

    hidden_error = output_error.dot(W2.T) * tanh_derivative(hidden_input)
    W1_gradient = X.T.dot(hidden_error) / X.shape[0]
    b1_gradient = np.mean(hidden_error, axis=0)

    W2 -= learning_rate * W2_gradient
    b2 -= learning_rate * b2_gradient
    W1 -= learning_rate * W1_gradient
    b1 -= learning_rate * b1_gradient

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
plt.plot(X, y_target, label="Target Output", color="blue")
plt.plot(X, y_pred, label="MLP Approximation", color="red", linestyle="--")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.show()
