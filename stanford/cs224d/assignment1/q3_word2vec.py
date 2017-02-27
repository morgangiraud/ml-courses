import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length
    
    ### YOUR CODE HERE
    x /= np.sqrt(np.sum(x * x, axis=1)).reshape(x.shape[0], 1) + 1e-30
    ### END YOUR CODE
    
    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]])) 
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert(x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""

def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """
    
    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, assuming the softmax prediction function and cross      
    # entropy loss.                                                   
    
    # Inputs:                                                         
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in 
    #   the written component or \hat{r} in an earlier version)
    # - target: integer, the index of the target word               
    # - outputVectors: "output" vectors (as rows) for all tokens     
    # - dataset: needed for negative sampling, unused here.         
    
    # Outputs:                                                        
    # - cost: cross entropy cost for the softmax word prediction    
    # - gradPred: the gradient with respect to the predicted word   
    #        vector                                                
    # - grad: the gradient with respect to all the other word        
    #        vectors                                               
    
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!                                                  
    
    ### YOUR CODE HERE
    # print('predicted', predicted.shape)
    # print('target', target)
    # print('outputVectors', outputVectors.shape)
    # outputVectors: W x n

    # We make the final predictions
    y_hat = softmax(outputVectors.dot(predicted)) # y_hat: (W,)
    # We compare to reality and get the cost (distance from truth)
    cost = - np.log(y_hat[target])

    y_hat[target] -= 1 
    gradPred = outputVectors.T.dot(y_hat).flatten() # gradPred: n x 1 -> (n,)
    # print('gradPred', gradPred.shape)

    # in Numpy you can't do an outer product of two one dim tensor, they must be reshapeed first
    grad = y_hat.reshape((y_hat.shape[0], 1)).dot(predicted.reshape((1, predicted.shape[0]))) # W x n
    # print('grad', grad.shape)

    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, 
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector  
    # and one target word vector as a building block for word2vec     
    # models, using the negative sampling technique. K is the sample  
    # size. You might want to use dataset.sampleTokenIdx() to sample  
    # a random word index. 
    # 
    # Note: See test_word2vec below for dataset's initialization.
    #                                       
    # Input/Output Specifications: same as softmaxCostAndGradient     
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!
    
    ### YOUR CODE HERE
    # print('predicted', predicted.shape) # (n,)
    # print('target', target)
    # print('outputVectors', outputVectors.shape) # (W, n)

    U_idx = [target]
    for i in range(K):
        idx = dataset.sampleTokenIdx()
        while idx == target:
            idx = dataset.sampleTokenIdx()
        U_idx.append(idx)

    U = outputVectors[U_idx,:] # (K+1, n)
    labels = np.array([1] + [-1 for k in range(K)]) # (K+1,)

    sig = sigmoid(U.dot(predicted) * labels) # (K+1,)
    cost = - np.sum(np.log(sig))

    gradPred = U.T.dot((sig - 1) * labels) # n x 1
    gradU = ((sig - 1) * labels).reshape((K+1, 1)).dot(
        predicted.reshape((1, predicted.shape[0]))
    ) # (K+1, n)
    # We have to make the gradient flow to the original output vectors
    # Since it is just a selection of vectors, the gradient flow unchanged
    grad = np.zeros(outputVectors.shape)
    for i in range(K+1): 
        grad[U_idx[i]] += gradU[i, :] # You do += because stistically you might have an indices multiple times

    
    ### END YOUR CODE
    
    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:                                                         
    # - currrentWord: a string of the current center word           
    # - C: integer, context size                                    
    # - contextWords: list of no more than 2*C strings, the context words                                               
    # - tokens: a dictionary that maps words to their indices in    
    #      the word vector list                                
    # - inputVectors: "input" word vectors (as rows) for all tokens           
    # - outputVectors: "output" word vectors (as rows) for all tokens         
    # - word2vecCostAndGradient: the cost and gradient function for 
    #      a prediction vector given the target word vectors,  
    #      could be one of the two cost functions you          
    #      implemented above

    # Outputs:                                                        
    # - cost: the cost function value for the skip-gram model       
    # - grad: the gradient with respect to the word vectors         
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    ### YOUR CODE HERE
    # inputVectors: W x n
    # outputVectors: W x n
    current_idx = tokens[currentWord]
    predicted = inputVectors[current_idx, :]

    cost = 0.
    gradInputVectors = np.zeros(inputVectors.shape)
    gradOutputVectors = np.zeros(outputVectors.shape)
    # print('gradInputVectors', gradInputVectors.shape)
    # print('gradOutputVectors', gradInputVectors.shape)
    for word in contextWords:
        idx = tokens[word]
        c, gradPred, grad = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
        cost += c
        # print('gradPred', gradPred.shape)
        # print('grad', grad.shape)
        # print('current_idx', current_idx)
        gradInputVectors[current_idx, :] += gradPred
        gradOutputVectors += grad
    
    ### END YOUR CODE
    
    return cost, gradInputVectors, gradOutputVectors

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.            
    # Input/Output specifications: same as the skip-gram model        
    # We will not provide starter code for this function, but feel    
    # free to reference the code you previously wrote for this        
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #  
    #################################################################
    
    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    # We have to predict
    p_idx = [tokens[word] for word in contextWords]
    predicted = np.sum(inputVectors[p_idx, :], axis=0)

    idx = tokens[currentWord] 
    cost, gradPred, gradOut = word2vecCostAndGradient(predicted, idx, outputVectors, dataset)
    # We make the gradient flow back the original input vectors
    gradIn = np.zeros(inputVectors.shape)
    for idx in p_idx:
        gradIn[idx, :] += gradPred
    ### END YOUR CODE
    
    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
