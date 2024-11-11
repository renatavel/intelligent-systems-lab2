import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1, 1, 10)  
y = np.linspace(-1, 1, 10) 
X, Y = np.meshgrid(x, y)    
Z = np.sin(np.pi * X) * np.cos(np.pi * Y)  


X_train = np.column_stack((X.ravel(), Y.ravel()))  
y_train = Z.ravel().reshape(-1, 1)                 

input_size = 2    
hidden_size = 6  
output_size = 1   
learning_rate = 0.1

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
    hidden_input = X_train.dot(W1) + b1  
    hidden_output = tanh(hidden_input)   

    final_input = hidden_output.dot(W2) + b2  
    y_pred = linear(final_input)             

    loss = np.mean((y_pred - y_train)**2)


    output_error = (y_pred - y_train) * linear_derivative(final_input)
    W2_gradient = hidden_output.T.dot(output_error) / X_train.shape[0]
    b2_gradient = np.mean(output_error, axis=0)

    hidden_error = output_error.dot(W2.T) * tanh_derivative(hidden_input)
    W1_gradient = X_train.T.dot(hidden_error) / X_train.shape[0]
    b1_gradient = np.mean(hidden_error, axis=0)

    W2 -= learning_rate * W2_gradient
    b2 -= learning_rate * b2_gradient
    W1 -= learning_rate * W1_gradient
    b1 -= learning_rate * b1_gradient

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

y_pred = y_pred.reshape(X.shape)  

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap="viridis")
ax1.set_title("Target Surface")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, y_pred, cmap="viridis")
ax2.set_title("MLP Approximated Surface")

plt.show()