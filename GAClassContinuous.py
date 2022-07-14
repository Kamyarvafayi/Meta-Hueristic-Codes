import numpy as np
# In[]: For sort
a = np.array([[10,5,8],[7,2,10],[4,3,5]])
a = a[a[:, 0].argsort()]
# In[]: Genetic Algorithm
class Ga_Algorithm_Continuous:
    def __init__(self,Iter_Number = 10000):
        self.Iter_Number = Iter_Number
    def Initial_Solutions(self):
        Initial_Solution = np.array([np.random.uniform(self.Low, self.High, size = np.size(self.Obj_Weights)) for i in range(self.Population_Size)])
        return Initial_Solution
    def Continuous_CrossOver(self, Sample1, Sample2):
        Gamma = 0.1
        RandomAlpha = np.random.randint(-1,1)*Gamma + np.random.rand(np.shape(self.Obj_Weights)[0])
        New_Solution1 = RandomAlpha*Sample1+(1-RandomAlpha)*Sample2
        New_Solution2 = RandomAlpha*Sample2+(1-RandomAlpha)*Sample1
        for i in range(np.shape(self.Obj_Weights)[0]):
            if New_Solution1[i]>= self.High:
               New_Solution1[i] = self.High
            elif New_Solution1[i]<= self.Low:
               New_Solution1[i] = self.High
            if New_Solution2[i]>= self.High:
               New_Solution2[i] = self.High
            elif New_Solution2[i]<= self.Low:
               New_Solution2[i] = self.High
        return New_Solution1, New_Solution2
    def Continuous_Mutation(self, Sample, Mu_Coef):
        Random = np.random.randint(0,np.shape(self.Obj_Weights)[0], size = int(self.Obj_Weights.shape[0]/Mu_Coef))
        Normal_Mu = self.High/10
        New_solution = Sample.copy()
        for i in range(int(self.Obj_Weights.shape[0]/Mu_Coef)):
            Random2 = New_solution[Random[i]] + Normal_Mu*np.random.randn()
            if Random2 >= self.High:
                New_solution[Random[i]] = self.High
            elif Random2 <= self.Low:
                New_solution[Random[i]] = self.Low
            else:
                New_solution[Random[i]] = Random2
        return New_solution.reshape(1,New_solution.shape[0])   
    def Find_Objective(self, New_Solution):
        Objective = np.dot(self.Obj_Weights, New_Solution)
        Constriant = np.array([np.dot(self.Const_Weights[i], New_Solution) for i in range(self.Const_Weights.shape[0])])
        Violation = Constriant-self.b
        return Objective + self.C*np.sum(Violation[Violation>0])
    def Fit_GA(self, Objective_Function, Constraint = np.array([0]), b = np.array([0]), C = 100, lower_bound = 0, upper_bound = 1, Population_Size = 10):
        self.Objectives = []
        self.All_Possible_Solutions = []
        self.Obj_Weights = Objective_Function
        self.b = b
        self.C = C
        self.Population_Size = Population_Size
        self.Const_Weights = Constraint
        self.Low = lower_bound
        self.High = upper_bound
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
                self.Population = np.append(self.Population,self.Continuous_Mutation(Mutation_solution[j], np.shape(self.Obj_Weights)[0]/4), axis=0)
        # Removing Duplicates from Population    
            self.Population = np.unique(np.round(self.Population, decimals = 3), axis=0)        
        # Find Objective
            self.Objectives = np.array([self.Find_Objective(self.Population[i]) for i in range(self.Population.shape[0])]) 
            self.Population = self.Population[self.Objectives.argsort()]
            self.Objectives_Sorted = np.sort(self.Objectives)
            self.Population = self.Population[0:self.Population_Size,:]
            self.Objectives_Sorted = self.Objectives_Sorted[0:self.Population_Size]
            print(self.Objectives_Sorted[0])
# In[]: without constraints
GA = Ga_Algorithm_Continuous(Iter_Number = 5000)  
Obj = np.random.randint(-40,5,size = 100)           
Ga = GA.Fit_GA(Obj , lower_bound = 0, upper_bound = 10, Population_Size = 10)
Objectives = GA.Objectives_Sorted
Solutions = GA.Population
# In[]: Constraint problem 
np.random.seed(seed=1)       
Obj2 = np.random.randint(-40,5,size = 10) 
Cons = np.array([np.random.randint(0,10,size = 10) for j in range(10)])
b = np.array([120, 100, 80, 80, 90,120, 110, 90, 90, 110]) 
c = 100 
GA2 = Ga_Algorithm_Continuous(Iter_Number = 5000)          
Ga2 = GA2.Fit_GA(Obj2 , Cons, b, c, lower_bound = 0, upper_bound = 10, Population_Size = 15)
Objectives2 = GA2.Objectives_Sorted
Solutions2 = GA2.Population
Violation =  np.dot(Cons, Solutions2[0]) - b   