import numpy as np

# One-hot encoder
def one_hot_encoder(y):
    """
    Assumes label of size (batch_size, 1)
    """
    # Get the number of samples
    batch_size = y.shape[0]

    # Reshape (batch_size, 1) to 1-d vector
    y = y.reshape(batch_size)

    # How many zeroes?
    n_values = np.max(y) + 1

    # Initilize zeroes and assign ones
    one_hots = np.zeros((batch_size, n_values))
    one_hots[np.arange(batch_size), y] = 1

    return one_hots