import numpy as np
# In[]: Genetic Algorithm
class Ga_Algorithm_Continuous:
    def __init__(self,Iter_Number = 10000):
        self.Iter_Number = Iter_Number
    def Initial_Solutions(self):
        Initial_Solution = np.array([np.random.uniform(self.Low, self.High, size = np.shape(self.Input)[1]) for i in range(self.Population_Size)])
        return Initial_Solution
    def Continuous_CrossOver(self, Sample1, Sample2):
        Gamma = 0.1
        RandomAlpha = np.random.randint(-1,1)*Gamma + np.random.rand(np.shape(self.Input)[1])
        New_Solution1 = RandomAlpha*Sample1+(1-RandomAlpha)*Sample2
        New_Solution2 = RandomAlpha*Sample2+(1-RandomAlpha)*Sample1
        return New_Solution1, New_Solution2
    def Continuous_Mutation(self, Sample, Mu_Coef):
        Random = np.random.randint(0,np.shape(self.Input)[1], size = int((self.Input.shape[1])/Mu_Coef))
        Normal_Mu = self.High/10
        New_solution = Sample.copy()
        for i in range(int((self.Input.shape[1])/Mu_Coef)):
            Random2 = New_solution[Random[i]] + Normal_Mu*np.random.randn()
            New_solution[Random[i]] = Random2
        return New_solution.reshape(1,New_solution.shape[0])   
    def Find_Objective(self, New_Solution):
        Linear_Objective = np.zeros([self.Input.shape[0],1])
        Sigmoid_Objective = np.zeros([self.Input.shape[0],1])
        error = np.zeros([self.Input.shape[0],1])
        for i in range(self.Input.shape[0]):
            Linear_Objective[i] = (self.Input[i]@New_Solution)
            Sigmoid_Objective[i] = 1/(np.exp(-Linear_Objective[i])+1)
            error[i] = -self.Target[i]*(np.log(Sigmoid_Objective[i])) + -(1-self.Target[i])*(np.log(1-Sigmoid_Objective[i]))
        return np.sum(error) + (1/self.L2)*np.sum(np.abs(New_Solution))
    def Predict(self, Sample):
        Final_Weights = self.Population[0].reshape(1,self.Input.shape[1])
        Value = np.round(1/(np.exp(-Final_Weights.dot(Sample.T))+1))
        return Value
    def Fit_GA(self, Input, Target, L2 = 1, Population_Size = 10):
        self.Objectives = []
        self.All_Possible_Solutions = []
        self.Input = np.append(Input, np.ones([Input.shape[0], 1]), axis = 1)
        self.Target = Target
        self.L2 = L2
        self.Population_Size = Population_Size
        self.Low = -10
        self.High = 10
        self.Initial_solution = self.Initial_Solutions()
        self.All_Possible_Solutions.append(self.Initial_solution)
        self.Objectives = np.array([self.Find_Objective(self.Initial_solution[i]) for i in range(self.Initial_solution.shape[0])])
        self.Population = self.Initial_solution[self.Objectives.argsort()]
        self.Objectives_Sorted = np.sort(self.Objectives)
        for i in range(self.Iter_Number):
            CRoss_solutions = self.Population.copy()
            Mutation_solution = self.Population.copy()
        # Crossover
            for j in range(self.Population_Size):
                for k in range(self.Population_Size):
                    self.Population = np.append(self.Population, self.Continuous_CrossOver(CRoss_solutions[j],CRoss_solutions[k]), axis=0)               
        # Mutation 
            for j in range(self.Population_Size):
                self.Population = np.append(self.Population,self.Continuous_Mutation(Mutation_solution[j], 1), axis=0)
        # Removing Duplicates from Population    
            self.Population = np.unique(np.round(self.Population, decimals = 3), axis=0)        
        # Find Objective
            self.Objectives = np.array([self.Find_Objective(self.Population[i]) for i in range(self.Population.shape[0])]) 
            self.Population = self.Population[self.Objectives.argsort()]
            self.Objectives_Sorted = np.sort(self.Objectives)
            self.Population = self.Population[0:self.Population_Size,:]
            self.Objectives_Sorted = self.Objectives_Sorted[0:self.Population_Size]
            print(self.Objectives_Sorted[0])
# In[]: Moons Dataset
from sklearn import datasets
Input , Target = datasets.make_moons(n_samples=1000, shuffle=True, noise=0.0)
Ga2 = Ga_Algorithm_Continuous(100)
GA2 = Ga2.Fit_GA(Input[0:700], Target[0:700].reshape(Target[0:700].shape[0],1), Population_Size = 10, L2 = 10)   
Input2 = np.append(Input[700:], np.ones([300, 1]), axis = 1)
Objectives2 = Ga2.Objectives_Sorted
Solutions2 = Ga2.Population
Predicted = Ga2.Predict(Input2)
error = Predicted.T - Target[700:].reshape(Target[700:].shape[0],1)
print(np.sum(np.abs(error)))
# In[]:
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, penalty='l2', C = 10).fit(Input[0:700], Target[0:700].reshape(Target[0:700].shape[0],1))
print(clf.intercept_)
print(clf.coef_)
sk_Predicted=clf.predict(Input[700:])
