import numpy as np
import matplotlib.pyplot as plt
import mnist
from activation import relu, sigmoid, sigmoid_prime, softmax
from helper import one_hot_encoder
from initializer import initialize_weight
from utils import dataloader

def train(train_x, train_y, learning_rate=0.1, num_epochs=50, batch_size=1):
    # Flatten input (num_samples, 28, 28) -> (num_samples, 784) 
    x = train_x.reshape(train_x.shape[0], -1)
    num_samples = x.shape[0]
    
    # Turn labels into their one-hot representations
    y = one_hot_encoder(train_y)

    # Make a data loader
    trainloader = dataloader(x, y, batch_size=batch_size, shuffle=True)
    
    # Initialize weights
    w1, b1 = initialize_weight((784, 256), bias=True)
    w2, b2 = initialize_weight((256, 10), bias=True)

    loss_history = []
    for epoch in range(1, num_epochs+1):
        print("Epoch {}/{}\n===============".format(epoch, num_epochs))

        batch_loss = 0
        acc = 0
        for inputs, labels in trainloader:
            # Number of samples per batch
            m = inputs.shape[0]
            
            # Forward Prop
            h1 = np.dot(inputs, w1) + b1
            a1 = sigmoid(h1)
            h2 = np.dot(a1, w2) + b2
            a2 = softmax(h2)
            out = a2

            # Cross Entropy Loss
            batch_loss += cross_entropy_loss(out, labels.argmax(axis=1).reshape(m,1))

            # Compute Accuracy
            pred = np.argmax(out, axis=1)
            pred = pred.reshape(pred.shape[0], 1)
            acc += np.sum(pred == labels.argmax(axis=1).reshape(m,1))

            # Backward Prop
            dh2 = a2 - labels 
            dw2 = (1/m) * np.dot(a1.T, dh2)
            db2 = (1/m) * np.sum(dh2, axis=0, keepdims=True)

            dh1 = np.dot(dh2, w2.T) * sigmoid_prime(a1)
            dw1 = (1/m) * np.dot(inputs.T, dh1)
            db1 = (1/m) * np.sum(dh1, axis=0, keepdims=True)

            # Weight (and bias) update
            w1 -= learning_rate * dw1
            b1 -= learning_rate * db1
            w2 -= learning_rate * dw2
            b2 -= learning_rate * db2
            
        loss_history.append(batch_loss/num_samples)
        print("Loss: {:.6f}".format(batch_loss/num_samples))
        print("Accuracy: {:.2f}%\n".format(acc/num_samples*100))

    return w1, b1, w2, b2, loss_history

def cross_entropy_loss(out, y):
    batch_size = y.shape[0]
    y = y.reshape(batch_size)
    log_likelihood = -np.log(out[np.arange(batch_size), y])
    return np.sum(log_likelihood)

def test(test_x, test_y, w1, b1, w2, b2):
    # Flatten input (batch_size, 28, 28) -> (batch_size, 784) 
    x = test_x.reshape(test_x.shape[0], -1)
    m = x.shape[0]
    
    # Turn labels into their one-hot representations
    y = one_hot_encoder(test_y)

    # Forward Pass
    h1 = np.dot(x, w1) + b1
    a1 = sigmoid(h1)
    h2 = np.dot(a1, w2) + b2
    a2 = softmax(h2)
    out = a2

    # Cross Entropy Loss
    loss = cross_entropy_loss(out, test_y)
    print("Loss: {:.6f}".format(loss/m))

    # Compute and print accuracy
    pred = np.argmax(out, axis=1)
    pred = pred.reshape(pred.shape[0], 1)
    acc = np.mean(pred == test_y)
    print("Accuracy: {:.2f}%\n".format(acc*100))

def main():
    # Load dataset
    train_x, train_y = mnist.load_dataset(download=True, train=True)
    test_x, test_y = mnist.load_dataset(download=True, train=False)

    # Batch Gradient Descent
    print("Training using Batch Gradient Descent")
    w1_bgd, b1_bgd, w2_bgd, b2_bgd, loss_history_bgd = train(train_x, train_y, learning_rate=0.2, num_epochs=20, batch_size=None)

    # Minibatch Gradient Descent with batch_size 64
    print("Training using mini-Batch Gradient Descent with batch size of 64")
    w1_mbgd, b1_mbgd, w2_mbgd, b2_mbgd, loss_history_mbgd = train(train_x, train_y, learning_rate=0.1, num_epochs=20, batch_size=64)

    # Display loss curve
    plt.plot(loss_history_bgd, label="Batch Gradient Descent")
    plt.plot(loss_history_mbgd, label="mini-Batch Gradient Descent")
    plt.legend()
    plt.show()

    # Test
    print("Batch Gradient Descent")
    test(test_x, test_y, w1_bgd, b1_bgd, w2_bgd, b2_bgd)
    print("mini-Batch Gradient Descent")
    test(test_x, test_y, w1_mbgd, b1_mbgd, w2_mbgd, b2_mbgd)
    
main()