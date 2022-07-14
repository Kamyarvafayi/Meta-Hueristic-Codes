import numpy as np
import random
# In[]:
class Ga_Optimizer_ANN:
    def __init__(self, Iter = 1000, Population_size = 5, Input_Layer = 3, Hidden_Layer = 5, Output_Layer = 1, Out_Put_Activation = "linear", Hidden_Activation = "tanh"):
        self.Iter = Iter
        self.Population_size = Population_size
        self.Input_Layer = Input_Layer
        self.Hidden_Layer = Hidden_Layer
        self.Output_Layer = Output_Layer
        self.Out_Activation = Out_Put_Activation
        self.Hidden_Activation = Hidden_Activation
    def Initialization(self):
        self.Initial_Solution_Dim = (self.Input_Layer+1)*self.Hidden_Layer + (self.Hidden_Layer+1)*self.Output_Layer
        self.Initial_Sol = np.array([np.random.uniform(low = -1, high = 1, size = self.Initial_Solution_Dim) for i in range(self.Population_size)])
        #self.Initial_Sol = np.array([[0.0 for j in range(self.Initial_Solution_Dim)] for i in range(self.Population_size)])
    def CrossOver(self, Sample1, Sample2):
        gamma = 0.1
        Random_Neuron_Input_Layer = np.random.randint(0,self.Hidden_Layer)
        Random_Neuron_Hidden_Layer = np.random.randint(0,self.Output_Layer)
        rand = np.random.rand()
        Random_Coeff = np.random.randint(-1,1)*gamma + np.random.uniform(0,1,size = self.Initial_Solution_Dim)
        New_Solution1 = Sample1.copy()
        New_Solution2 = Sample2.copy()
        if rand<0.7:
            New_Solution1[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)] = Random_Coeff[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)]*Sample1[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)]+(1-Random_Coeff[Random_Neuron_Input_Layer*(self.Input_Layer+1):(Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)])*Sample2[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)]
            New_Solution2[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)] = Random_Coeff[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)]*Sample2[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)]+(1-Random_Coeff[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)])*Sample1[Random_Neuron_Input_Layer*(self.Input_Layer+1): (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)]
            return New_Solution1, New_Solution2
        else:
            New_Solution1[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)] = Random_Coeff[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)]*Sample1[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)]+(1-Random_Coeff[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)])*Sample2[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)]
            New_Solution2[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)] = Random_Coeff[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)]*Sample2[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)]+(1-Random_Coeff[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)])*Sample1[(self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1):(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1)]
            return New_Solution1, New_Solution2
    def Mutation(self, Sample):
        step = 5
        Random_Neuron_Input_Layer = np.random.randint(0,self.Hidden_Layer)
        Random_Neuron_Hidden_Layer = np.random.randint(0,self.Output_Layer)
        #Mutation_Dim = np.max([1,int(np.ceil(self.Initial_Solution_Dim/10))])
        #Random_Indices = random.sample(range(self.Initial_Solution_Dim),Mutation_Dim)
        New_Solution = Sample.copy()
        rand = np.random.rand()
        if rand<0.7:
            for Indices in range(Random_Neuron_Input_Layer*(self.Input_Layer+1), (Random_Neuron_Input_Layer+1)*(self.Input_Layer+1)):
                New_Solution[Indices] = Sample[Indices] + np.random.randn()*step
            return New_Solution.reshape(1,New_Solution.shape[0])
        else:
            for Indices in range((self.Input_Layer+1)*self.Hidden_Layer+Random_Neuron_Hidden_Layer*(self.Output_Layer+1),(self.Input_Layer+1)*self.Hidden_Layer+(Random_Neuron_Hidden_Layer+1)*(self.Output_Layer+1) ):
                New_Solution[Indices] = Sample[Indices] + np.random.randn()*step
            return New_Solution.reshape(1,New_Solution.shape[0])
    def Sigmoid(self,x):
        return np.tanh(x) 
        return x**2
    def Find_Objective(self, Solution):
        errors = 0
        for k in range(self.Input.shape[0]):    
            Hidden_Layers_Output = []
            for i in range(self.Hidden_Layer):
                Hidden_Layers_Output.append(Solution[i*(self.Input_Layer+1):(i+1)*(self.Input_Layer+1)].dot(self.Input[k]))
                if self.Hidden_Activation =="tanh":
                    Hidden_Layers_Output[i] = self.Sigmoid(Hidden_Layers_Output[i])
            Output_Layers_Output = []
            for j in range(self.Output_Layer):
                Output_Layers_Output.append(np.array([Hidden_Layers_Output]).dot(Solution[(self.Input_Layer+1)*self.Hidden_Layer + j*self.Hidden_Layer: (self.Input_Layer+1)*self.Hidden_Layer + (j+1)*self.Hidden_Layer]))
                if self.Out_Activation =="tanh":
                    Output_Layers_Output[j] = self.Sigmoid(Output_Layers_Output[j])
                errors += np.abs(self.Target[k,j] - Output_Layers_Output[j])
        return np.sum(errors) + (1/self.L2)*np.sum(np.abs(Solution)) + (1/self.L2)*np.sum(np.square(Solution))
    def Predict(self, X):
        Hidden_Layers_Output = []
        for i in range(self.Hidden_Layer):
            Hidden_Layers_Output.append(self.Population[0][i*(self.Input_Layer+1):(i+1)*(self.Input_Layer+1)].dot(X))
            if self.Hidden_Activation =="tanh":
                Hidden_Layers_Output[i] = self.Sigmoid(Hidden_Layers_Output[i])
        Output_Layers_Output = []
        for j in range(self.Output_Layer):
            Output_Layers_Output.append(np.array([Hidden_Layers_Output]).dot(self.Population[0][(self.Input_Layer+1)*self.Hidden_Layer + j*self.Hidden_Layer: (self.Input_Layer+1)*self.Hidden_Layer + (j+1)*self.Hidden_Layer]))
            if self.Out_Activation =="tanh":
                Output_Layers_Output[j] = self.Sigmoid(Output_Layers_Output[j])
        return Output_Layers_Output 
    def Ann_Fit_Ga(self, Input, Target, L1=1, L2=1):
        self.Objectives = []
        self.All_Possible_Solutions = []
        self.Input = np.append(Input, np.ones([Input.shape[0], 1]), axis = 1)
        self.Target = Target
        self.L1 = L1
        self.L2 = L2
        self.Initialization()
        self.All_Possible_Solutions.append(self.Initial_Sol)
        self.Objectives = np.array([self.Find_Objective(self.Initial_Sol[i]) for i in range(self.Initial_Sol.shape[0])])
        self.Population = self.Initial_Sol[self.Objectives.argsort(axis = 0)]
        self.Objectives_Sorted = np.sort(self.Objectives)
        for i in range(self.Iter):
            CRoss_solutions = self.Population.copy()
            Mutation_solution = self.Population.copy()
            for k in range(10):
        # Crossover
                for j in range(self.Population_size):
                    for k in range(self.Population_size):
                        self.Population = np.append(self.Population, self.CrossOver(CRoss_solutions[j],CRoss_solutions[k]), axis=0)               
        # Mutation 
                for j in range(self.Population_size):
                    self.Population = np.append(self.Population,self.Mutation(Mutation_solution[j]), axis=0)
        # Removing Duplicates from Population    
            self.Population = np.unique(np.round(self.Population, decimals = 3), axis=0) 
        # Find Objective
            self.Objectives = np.array([self.Find_Objective(self.Population[i]) for i in range(self.Population.shape[0])]) 
            self.Population = self.Population[self.Objectives.argsort()]
            self.Objectives_Sorted = np.sort(self.Objectives)
            self.Population = self.Population[0:self.Population_size,:]
            self.Objectives_Sorted = self.Objectives_Sorted[0:self.Population_size]
            print(self.Objectives_Sorted[0])
# In[]:
x1 = np.array([np.linspace(-5, 5, 500)])
x2 = np.array([np.linspace(0, 5, 500)])
x = np.append(x1,x2,axis=0)
print(x)
y = 7.5*(1/(np.exp(-2*x1+x2)+x1)) - 10 #+ np.random.randn(200)
y2 = np.sin(x)
ANN = Ga_Optimizer_ANN(Iter =100, Input_Layer = 2, Hidden_Layer = 3, Population_size= 10, Out_Put_Activation = "linear", Hidden_Activation = "tanh")
Model = ANN.Ann_Fit_Ga(x.T, y.T, L1 = 1, L2 = 10)
Weights = ANN.Population[-1]
error = ANN.Find_Objective(ANN.Population[-1])
print(ANN.Predict(np.array([0,5,1])))
np.sum(np.abs(Weights)) + np.sum(np.square(Weights))/10


# In[]:
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn.datasets as sk
import sklearn.model_selection
from keras.utils import to_categorical
# In
# moonsdata, moonsclass = sk.make_moons(10000, noise = 0.15)
# Train_X, Test_X, Train_Y, Test_Y = sklearn.model_selection.train_test_split(moonsdata, moonsclass)
# fig1 = plt.figure(figsize= [10,10])
# plt.scatter(Train_X[Train_Y == 0,0], Train_X[Train_Y == 0,1])
# plt.scatter(Train_X[Train_Y == 1,0], Train_X[Train_Y == 1,1])
# plt.legend(['First Class','Second Class'])
# plt.grid()
# plt.figure(figsize= [10,10])
# plt.scatter(Test_X[Test_Y == 0,0], Test_X[Test_Y == 0,1])
# plt.scatter(Test_X[Test_Y == 1,0], Test_X[Test_Y == 1,1])
# plt.legend(['First Class','Second Class'])


model = keras.Sequential()
model.add(keras.layers.Input(shape = (2,)))
model.add(keras.layers.Dense(3, activation='tanh',kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
    bias_initializer=keras.initializers.Zeros()))
model.add(keras.layers.Dense(1, activation='linear',kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
    bias_initializer=keras.initializers.Zeros()))
model.compile(optimizer='sgd', loss=keras.losses.MeanSquaredError())
# This builds the model for the first time:
model.fit(x.T, y.T, batch_size=25, epochs=100)
