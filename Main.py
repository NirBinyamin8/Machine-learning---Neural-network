
from sklearn.model_selection import train_test_split
import numpy as np
from NeuralNetwork import NeuralNetwork
from NeuralNetwork import NeuralNetwork_2
import pickle

# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def relu(x):
    return np.maximum(0, x)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

#Load the data
def load_data():
    # Load the MNIST dataset
    X = np.load('MNIST-data.npy')
    X = X.reshape(X.shape[0], -1)
    y = np.load("MNIST-lables.npy")
    y = np.eye(10)[y]
    # Split the data into train, validation, and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    return x_train, x_valid,x_test, y_train,y_valid, y_test


#Experiments
def Experiments(x_train,y_train,x_valid,y_valid):
    Active_Func = [(sigmoid, sigmoid_derivative), (relu, relu_derivative)]
    Epochs = [50, 100]
    hidden_dim_1 = [400, 800, 1000]
    Lr = [0.01, 0.1]
    l2_lambda_values = [0.01, 0.1]
    best_model_3 = None
    best_score_3 = 0
    i = 1

    print ("Start the experiment for 3 layers model :")
    # 3 layers net
    for func in Active_Func:
        for epochs in Epochs:
            print("epoches = ",epochs)
            for dim in hidden_dim_1:
                for lr in Lr:
                    for l in l2_lambda_values:
                        model = NeuralNetwork(hidden_dim=dim, activation_function=func[0],
                                              activation_function_deriavte=func[1], epochs=epochs, learning_rate=lr,
                                              lamda=l)
                        model.fit(x_train, y_train)
                        score = model.score(x_valid, y_valid)
                        print("model ", i, "score = ", score)
                        i = i + 1
                        if score >= best_score_3:
                            best_model_3 = model
                            best_score_3 = score

    print("Start the experiment for 4 layers model :")
    # 4 layers models
    Epochs = [50]
    hidden_dim_1 = [400, 800]
    hidden_dim_2 = [100, 256]
    Lr = [0.01, 0.1]
    l2_lambda_values = [0.01, 0.1]
    best_model_4 = None
    best_score_4 = 0
    i = 1
    for func in Active_Func:
        for epochs in Epochs:
            for dim1 in hidden_dim_1:
                for dim2 in hidden_dim_2:
                    for lr in Lr:
                        for l in l2_lambda_values:
                            model = NeuralNetwork_2(hidden_dim1=dim1, hidden_dim2=dim2, activation_function=func[0],
                                                    activation_function_derivative=func[1], epochs=epochs,
                                                    learning_rate=lr,
                                                    lamda=l)
                            model.fit(x_train, y_train)
                            score = model.score(x_valid, y_valid)
                            print("model ", i, "score = ", score)
                            i = i + 1
                            if score >= best_score_4:
                                best_model_4 = model
                                best_score_4 = score

    if best_score_4 > best_score_3:
        Best_model = best_model_4
        archi=(4,Best_model.activation_function,Best_model.epochs,[Best_model.hidden_dim1,Best_model.hidden_dim2],Best_model.learning_rate,Best_model.l2_lambda)
        best_score=best_score_4
    else:
        Best_model = best_model_3
        archi = (3, Best_model.activation_function, Best_model.epochs, [Best_model.hidden_dim],
                 Best_model.learning_rate, Best_model.l2_lambda)
        best_score = best_score_3

    return archi,Best_model,best_score
def Examining_the_best_model(x_train,y_train,x_valid,y_valid,best_score,Best_model,archi):
    archi=list(archi)
    func=archi[1]
    if func =='Sigmoid':
        func=sigmoid
        derv_func=sigmoid_derivative
    else:
        func=relu
        derv_func=relu_derivative
    hidden_dim=archi[3]
    Lr=[0.1,0.01,0.001]
    epoch=[100,200,300]
    Lambada=[0.01,0.001]
    for e in epoch:
        for lr in Lr:
            for lambada in Lambada:
                print("epochs = ", e, "Lr = ", lr,"lambada = ",lambada)
                model = NeuralNetwork(epochs=e, hidden_dim=hidden_dim, lamda=lambada, activation_function=func,
                                      activation_function_deriavte=derv_func, learning_rate=lr)
                model.fit(x_train, y_train)
                score = model.score(x_valid, y_valid)
                print("The score is ", score, "**********")
                if score > best_score:
                    Best_model = model
                    best_score = score
                    archi[2] = e
                    archi[4] = lr
                    archi[5]=lambada

    Save(Best_model,archi)
    return archi, Best_model, best_score

#Save to file
def Save(model,archi):
    # save the model to disk
    filename = 'model.pkl'
    pickle.dump({'W1': model.W1, 'b1': model.b1, 'W2': model.W2, 'b2': model.b2,'archi': archi}, open(filename, 'wb'))
def load(filename = 'model.pkl'):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    model=NeuralNetwork()
    model.W1 = loaded_model['W1']
    model.b1 = loaded_model['b1']
    model.W2 = loaded_model['W2']
    model.b2 = loaded_model['b2']
    archi = loaded_model['archi']
    return archi,model


if __name__ == '__main__':

    #Load the data
    x_train, x_valid,x_test, y_train,y_valid, y_test=load_data()

    print("The model went through an experimental process in order to test the optimal hyperparameters, do you want to perform these experiments again or use the optimal model of the network? y/n")
    ans=input()

    if(ans=='y'):
        # First experiment
     archi,Best_model,Best_score=Experiments(x_train,y_train,x_valid,y_valid)
        # Second  experiment
     archi, model,Best_score = Examining_the_best_model(x_train, y_train, x_valid, y_valid, best_score=Best_score,
                                             Best_model=Best_model, archi=archi)
     score=Best_model.score(x_test,y_test)
     print("The score of our best model is : ",score)
     print("The architecture of out net is: "
     "Layers =",archi[0],"activation function = ",archi[1],"Number of epochs =",archi[2],"hidden dim = ",archi[3],"learning rate = ",archi[4],"l2 lambda = ",archi[5])
    else:
        #Load the best model from the file
        archi,Best_model = load()
        # Score on the test set
        score = Best_model.score(x_test, y_test)
        print("The score of our best model is : ", score)
        print("The architecture of our net is :"
              "Layers =", archi[0], ", activation function = ", archi[1], ", Number of epochs =", archi[2], ", hidden dim = ",
              archi[3], ", learning rate = ", archi[4], ", l2 lambda = ", archi[5])










