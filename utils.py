import numpy as np

def dataloader(x, y, batch_size=1, shuffle=False):
    """
    Dataloader for making minibatches
    Default is SGD i.e batch_size = 1
    """
    # Batch Gradient Descent
    if (batch_size==None):
        return [(x, y)]

    # Number of samples
    m = x.shape[0]

    # Shuffle dataset
    if (shuffle):
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation]
    
    # Count the number of minibatches
    num_batches = m // batch_size

    # Make Minibatches
    minibatches = []
    for i in range(num_batches):
        mb_x = x[i*batch_size: (i+1)*batch_size]
        mb_y = y[i*batch_size: (i+1)*batch_size]
        minibatches.append((mb_x, mb_y))

    if (num_batches * batch_size < m):
        mb_x = x[num_batches*batch_size:]
        mb_y = y[num_batches*batch_size:]
        minibatches.append((mb_x, mb_y))

    return minibatches