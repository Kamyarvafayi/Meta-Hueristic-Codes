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
        Random = np.random.randint(0,np.shape(self.Optimal_Solution)[0], size = 2)
        Random2 = np.random.randint(self.Low,self.High, size = 2)
        New_solution = self.Optimal_Solution.copy()
        New_solution[Random[0]], New_solution[Random[1]] = Random2[0], Random2[1]
        return New_solution[::-1]
    def Reverse(self):
        Random = np.random.randint(0,np.shape(self.Optimal_Solution)[0], size = 2)
        New_solution = self.Optimal_Solution.copy()
        flipped_Part = np.flip(New_solution[Random[0]:Random[1]])
        New_solution[Random[0]:Random[1]] = flipped_Part
        return New_solution
    def Find_Objective(self, New_Solution):
        Objective = np.dot(self.Obj_Weights, New_Solution)
        Constriant = np.dot(self.Const_Weights, New_Solution)
        return Objective + self.C*np.max(np.array([0,Constriant - self.b]))
    def Fit_SA(self, Objective_Function, Constraint, b, C = 100, lower_bound = 0, upper_bound = 5, Max_Iter = 10000):
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
                # else:
                #      rand = np.random.uniform(0,1)
                #      #if rand < np.exp(-1*DeltaObjective/self.temp):
                #      if rand < 2*np.exp(DeltaObjective*10/self.temp):    
                #          self.Optimal_Solution  = New_Solution
                #          self.Optimals.append(self.Optimal_Solution)
                #          self.Objectives.append(self.Find_Objective(self.Optimal_Solution))
                Iter += 1
            self.temp = self.temp*self.Alpha
            if self.temp < 0.01:
                end = True
# In[]: 
SA = SA_Algorithm(100,0.5)  
Obj = np.random.randint(-40,5,size = 20) 
Cons = np.random.randint(0,5,size = 20)
b = 70 
c = 100           
Sa = SA.Fit_SA(Obj , Cons, b, c, lower_bound = 0, upper_bound = 5, Max_Iter = 1000000)
objectives = SA.Objectives
Solutions = SA.Optimals
Constriant = np.dot(Cons, Solutions[0])
