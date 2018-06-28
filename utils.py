import numpy as np
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
# np.random.seed(42)

# Function to help us plot
def plot_points(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')
    
def plot_model(X, y, clf):
    plt.scatter(X[np.argwhere(y==0).flatten(),0],X[np.argwhere(y==0).flatten(),1],s = 50, color = 'red', edgecolor = 'k')
    plt.scatter(X[np.argwhere(y==1).flatten(),0],X[np.argwhere(y==1).flatten(),1],s = 50, color = 'cyan', edgecolor = 'k')

    minimum_x = min(X[:,0])
    maximum_x = max(X[:,0])
    minimum_y = min(X[:,1])
    maximum_y = max(X[:,1])
    minimum = min(minimum_x, minimum_y)
    maximum = min(maximum_x, maximum_y)
    
    plt.xlim(-0.05, 2.05)
    plt.ylim(-0.05, 2.05)
    plt.grid(False)
    plt.tick_params(
    axis='x',
    which='both',
    bottom='off',
    top='off')

    r = np.linspace(-0.04,2.05,300)
    s,t = np.meshgrid(r,r)
    s = np.reshape(s,(np.size(s),1))
    t = np.reshape(t,(np.size(t),1))
    h = np.concatenate((s,t),1)

    z = clf.predict(h)

    s.shape = (np.size(r),np.size(r))
    t.shape = (np.size(r),np.size(r))
    z.shape = (np.size(r),np.size(r))

    plt.contourf(s,t,z,colors = ['red','cyan'],alpha = 0.2,levels = range(-1,2))
    if len(np.unique(z)) > 1:
        plt.contour(s,t,z,colors = 'k', linewidths = 2)
    
def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 1000):
    import numpy as np
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
    return (-W[0]/W[1], -b/W[1])

def draw_dataset(X, y):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    #plt.xlim(-0.5,1.5)
    #plt.ylim(-0.5,1.5)
    for i in range(len(y)):
        if y[i]==0:
            plt.scatter(X[i][0], X[i][1], color='red', edgecolor = 'k')
        else:
            plt.scatter(X[i][0], X[i][1], color='cyan', edgecolor = 'k')

def display(m, b, color='g--', minimum=0, maximum=1):
    x = np.arange(minimum, maximum, 0.1)
    plt.plot(x, m*x+b, color)

def train_linear_model_manually(X, y):
    import matplotlib.pyplot as plt
    minimum = min(X[:,0])
    maximum = max(X[:,0])
    draw_dataset(X, y)
    boundary_line = trainPerceptronAlgorithm(X, y)
    solution_slope = boundary_line[0]
    solution_intercept = boundary_line[1]
    display(solution_slope, solution_intercept, 'k', minimum, maximum)
    plt.show()
    return

def train_linear_model(X, y, graph=True):
    import matplotlib.pyplot as plt
    minimum = min(X[:,0])
    maximum = max(X[:,0])
    draw_dataset(X, y)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    #coefs = model.coef_
    #m = coefs[0][0]
    #b = coefs[0][1]
    draw_dataset(X, y)
    #display(m, b, 'k', minimum, maximum)
    if graph:
        plot_model(X, y, model)
        plt.show()
    count = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            count += 1
    print(count,'correct out of',len(y))
    print('Accuracy:', count*1.0/len(y))
    return y_pred
    

def train_neural_network(X, y):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD
    from keras.utils import np_utils

    # Building the model
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(6,)))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(2, activation='softmax'))

    # Compiling the model
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(X, y, epochs=200, batch_size=100, verbose=0)

    score = model.evaluate(X, y)
    print("\n Training Accuracy:", score[1])
    
    predictions = model.predict(X)
    return np.argmax(predictions, axis=1)

def test_linear_model(features_train, targets_train, features_test, targets_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(features_train, targets_train)
    pred = model.predict(features_test)
    count = 0
    for i in range(len(targets_test)):
        if targets_test[i] == pred[i]:
            count += 1
    #print(count,'correct out of',len(targets_test))
    print('Accuracy:', count*1.0/len(targets_test))

def test_neural_network(features_train, targets_train, features_test, targets_test):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import SGD
    from keras.utils import np_utils

    # Building the model
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(6,)))
    model.add(Dropout(.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(2, activation='softmax'))

    # Compiling the model
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(features_train, targets_train, epochs=200, batch_size=100, verbose=0)

    score = model.evaluate(features_test, targets_test)
    print("\n Testing Accuracy:", score[1])