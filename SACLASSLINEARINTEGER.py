import numpy as np
class SA_Algorithm:
    def __init__(self,temperature, Alpha):
        self.temp = temperature
        self.Alpha = Alpha
    def Swap(self):
        Random = np.random.randint(0,np.shape(self.Optimal_Solution)[0], size = 2)
        New_solution = self.Optimal_Solution.copy()
        temp = New_solution[Random[0]]
        New_solution[Random[0]] = New_solution[Random[1]]
        New_solution[Random[1]] = temp
        return New_solution
    def Change(self):
        Random = np.random.randint(0,np.shape(self.Optimal_Solution)[0], size = int(self.Obj_Weights.shape[0]/10))
        Random2 = np.random.randint(self.Low,self.High, size = int(self.Obj_Weights.shape[0]/10))
        New_solution = self.Optimal_Solution.copy()
        for i in range(int(self.Obj_Weights.shape[0]/10)):
            New_solution[Random[i]] = Random2[i]
        return New_solution
    def Reverse(self):
        Random = np.random.randint(0,np.shape(self.Optimal_Solution)[0], size = 2)
        New_solution = self.Optimal_Solution.copy()
        flipped_Part = np.flip(New_solution[Random[0]:Random[1]])
        New_solution[Random[0]:Random[1]] = flipped_Part
        return New_solution
    def Find_Objective(self, New_Solution):
        Objective = np.dot(self.Obj_Weights, New_Solution)
        Constriant = np.array([np.dot(self.Const_Weights[i], New_Solution) for i in range(self.Const_Weights.shape[0])])
        Violation = Constriant-self.b
        return Objective + self.C*np.sum(Violation[Violation>0])
    def Fit_SA(self, Objective_Function, Constraint = np.array([0]), b = np.array([0]), C = 100, lower_bound = 0, upper_bound = 1, Max_Iter = 10000):
        self.Objectives = []
        self.Optimals = []
        self.Obj_Weights = Objective_Function
        self.b = b
        self.C = C
        self.Const_Weights = Constraint
        self.Low = lower_bound
        self.High = upper_bound + 1
        self.Optimal_Solution = np.random.randint(self.Low, self.High, size = np.size(self.Obj_Weights))
        self.Optimals.append(self.Optimal_Solution)
        self.Objectives.append(self.Find_Objective(self.Optimal_Solution))
        Iter = 0;
        end = False
        while end==False:
            while Iter<Max_Iter:
                rand = np.random.uniform(0,1)
                if rand < 0.75:
                    New_Solution = self.Change()
                elif rand < 0.90:
                    New_Solution = self.Reverse()
                else:
                    New_Solution = self.Swap()
                DeltaObjective = self.Find_Objective(self.Optimal_Solution)- self.Find_Objective(New_Solution)
                if DeltaObjective > 0:
                    self.Optimal_Solution  = New_Solution
                    self.Optimals.append(self.Optimal_Solution)
                    self.Objectives.append(self.Find_Objective(self.Optimal_Solution))
                else:
                      rand = np.random.uniform(0,1)
                      #if rand < np.exp(-1*DeltaObjective/self.temp):
                      if 2*rand < np.exp(DeltaObjective*10/self.temp) and DeltaObjective<0:    
                          self.Optimal_Solution  = New_Solution
                          self.Optimals.append(self.Optimal_Solution)
                          self.Objectives.append(self.Find_Objective(self.Optimal_Solution))
                Iter += 1
            self.temp = self.temp*self.Alpha
            if self.temp < 0.01:
                end = True
# In[]:
np.random.seed(seed=1)
SA = SA_Algorithm(100,0.5)  
Obj = np.random.randint(-40,5,size = 20) 
Cons = np.array([np.random.randint(0,5,size = 20) for j in range(10)])
b = np.array([70, 100, 60, 80, 90,120, 80, 70, 90, 110]) 
c = 100           
Sa = SA.Fit_SA(Obj , Cons, b, c, lower_bound = 0, upper_bound = 5, Max_Iter = 150000)
objectives = SA.Objectives
Solutions = SA.Optimals
Violation =  np.dot(Cons, Solutions[-1]) - b

# In[]:
SA2 = SA_Algorithm(100,0.5)  
Obj2 = np.random.randint(-40,5,size = 100)           
Sa2 = SA2.Fit_SA(Obj2 , lower_bound = 0, upper_bound = 5, Max_Iter = 300000)
objectives2 = SA2.Objectives
Solutions2 = SA2.Optimals