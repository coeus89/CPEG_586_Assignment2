import sys
from Utils import Utils
import numpy as np

def sigmoid(beta,X):
    return 1.0/(1.0 + np.exp(-np.dot(X, beta.T)))

def gradientBeta(beta, X, y):
    a = sigmoid(beta, X)
    part1 = a - y.reshape(X.shape[0], 1)
    grad = np.dot(part1.T, X)
    return grad

def logLoss(beta, X, y):
    a = sigmoid(beta,X)
    loss = -(y * np.log(a) + (1 - y) * np.log(1 - a))
    return np.sum(loss)

def trainUsingGradientDescent(X, y, beta, num_iter, alpha = 0.01):
    loss = logLoss(beta,X,y)
    for i in range(num_iter):
        beta = beta - (alpha * gradientBeta(beta,X,y))
        loss = logLoss(beta,X,y)
        if(i%10 == 0):
            print('iter = ' + str(i) + ' loss=' + str(loss))
    return beta

def classifyData(beta, X):
    a = sigmoid(beta, X)
    decision = np.where(a >= 0.5,1,0)
    return decision

def main():
    utils = Utils()

    # Load Data
    data = utils.readData("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\CPEG_586_Assignment2\\DataSet.csv")

    # or [:,;-1] normalize data - scale between 0-1
    X = utils.normalizeData(data[:,0:2])
    #print(X)

    # add 1's column to data
    X = np.hstack((np.ones((1,X.shape[0])).T, X))
    #print(X)

    Y = data[:,-1] #Expected Output, -1 means last column
    beta = np.zeros((1,X.shape[1])) # (1,3) in this example
    beta = trainUsingGradientDescent(X,Y,beta,1000) #optimize using gradient descent

    print("Logistic regression Model Coefficients: ", beta)

    y_predicted = classifyData(beta,X) #predictions from the trained model
    #print(y_predicted.shape)
    print("Number of correct predictions = ", str(np.sum(Y == y_predicted.reshape(Y.shape[0]))/len(X)*100) + "%")

    utils.plot_result(X, Y, beta) # plot results

if __name__ == "__main__":
    sys.exit(int(main() or 0))