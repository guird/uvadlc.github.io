import numpy as np

#softmax function written here for simplicity
def softmax(x):
    
    #inputs:
    #   x: an array of floats
    #outputs: 
    #   sofx: the softmax of x
    logp = np.zeros(x.shape)
    for i in range(x.shape[0]):
        
        logq = x[i,:]
        a = max(logq)
        logz = 0
        for q in logq:
            logz += np.exp(q - a)
        logz = a + np.log(logz)
        logp[i,:] = logq - logz
     
    return logp
