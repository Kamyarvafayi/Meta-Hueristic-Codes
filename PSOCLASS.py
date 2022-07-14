import numpy as np
class PSO_Algorithm:
    def __init__(self, Iteration = 10000, Swarm_Num = 10):
        self.Iteration = Iteration
        self.Swarm_Num = Swarm_Num
    def Initial_Positions(self):
        self.Initial_Position = np.array([np.random.uniform(self.Low, self.High, size = self.Obj_Weights.shape[0]) for i in range(self.Swarm_Num)])
        return self.Initial_Position.copy()
    def Find_Objective(self, New_Solution):
        Objective = np.dot(self.Obj_Weights, New_Solution)
        Constriant = np.array([np.dot(self.Const_Weights[i], New_Solution) for i in range(self.Const_Weights.shape[0])])
        Violation = Constriant-self.b
        print(Objective, "\n")
        print(Objective + self.C*np.sum(Violation[Violation>0]),"\n")
        print(self.C*np.sum(Violation[Violation>0]),"\n\n\n\n\n\n")
        return (Objective + self.C*np.sum(Violation[Violation>0]))
    def Find_New_Positions(self, Swarm_Index, r1, r2):
        self.V[Swarm_Index] = self.w * self.V[Swarm_Index] + r1*self.c1*(self.Swarm_Best_Positions[Swarm_Index]-self.Swarm_Positions[Swarm_Index])+r2*self.c2*(self.Global_Best_Position-self.Swarm_Positions[Swarm_Index])
        self.V[Swarm_Index][self.V[Swarm_Index]>np.round(self.High/10-self.Low/10)] = self.High/10-self.Low/10
        self.V[Swarm_Index][self.V[Swarm_Index]<-np.round(self.High/10-self.Low/10)] = -(self.High/10-self.Low/10)
        
        self.Swarm_Positions[Swarm_Index] = self.Swarm_Positions[Swarm_Index] + self.V[Swarm_Index]
        self.Swarm_Positions[Swarm_Index,self.Swarm_Positions[Swarm_Index]>self.High] = self.High 
        self.Swarm_Positions[Swarm_Index,self.Swarm_Positions[Swarm_Index]<self.Low] = self.Low 
        if self.Find_Objective(self.Swarm_Positions[Swarm_Index]) < self.Swarm_Best_Objectives[Swarm_Index] :
           self.Swarm_Best_Positions[Swarm_Index] = self.Swarm_Positions[Swarm_Index].copy() 
           self.Swarm_Best_Objectives[Swarm_Index] = self.Find_Objective(self.Swarm_Positions[Swarm_Index])
           if self.Swarm_Best_Objectives[Swarm_Index] < self.Global_Best_Objective:
               self.Global_Best_Objective = self.Swarm_Best_Objectives[Swarm_Index].copy()
               self.Global_Best_Position = self.Swarm_Best_Positions[Swarm_Index].copy()
    def Find_Global(self):
        self.Global_Best_Position = self.Swarm_Best_Positions[np.argmin(self.Swarm_Best_Objectives)]
        self.Global_Best_Objective = np.min(self.Swarm_Best_Objectives)
        print(self.Global_Best_Position,": ", self.Global_Best_Objective, "\n")
    def Fit_PSO(self, Objective_Function, Constraint = np.array([0]), b = np.array([0]), C = 100, lower_bound = 0, upper_bound = 1, w = 0.6, c1=2, c2=2):
        self.Obj_Weights = Objective_Function
        self.b = b
        self.C = C
        self.Const_Weights = Constraint
        self.Low = lower_bound
        self.High = upper_bound
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.Swarm_Positions = self.Initial_Positions().copy()
        self.Swarm_Best_Positions = self.Swarm_Positions 
        self.Swarm_Best_Objectives = self.Swarm_Objectives = np.array([self.Find_Objective(self.Swarm_Positions[i]) for i in range(self.Swarm_Num)])
        self.Global_Best_Position = self.Swarm_Best_Positions[np.argmin(self.Swarm_Best_Objectives)]
        self.Global_Best_Objective = np.min(self.Swarm_Best_Objectives)
        self.V = [np.random.uniform(-1,1, size = self.Obj_Weights.shape[0]) for i in range(self.Swarm_Num)]
        #self.V = [np.ones(shape=[1,self.Obj_Weights.shape[0]])*(self.High-self.Low)*0.1 for i in range(self.Swarm_Num)]
        Iter = 0
        while Iter < self.Iteration:
            for Swarm_Index in range(self.Swarm_Num):
                r1 = np.random.rand()
                r2 = np.random.rand()
                self.Find_New_Positions(Swarm_Index, r1, r2)
                #self.Find_Global()
            print(self.Global_Best_Position,": ", self.Global_Best_Objective)
            self.w = self.w*0.99
            Iter += 1
# In[]:
Obj = np.random.randint(-40,5,size = 10)             
PSO = PSO_Algorithm(300, 500)           
Pso = PSO.Fit_PSO(Obj , lower_bound = 0, upper_bound = 10, w = 1)
Positions = PSO.Swarm_Best_Positions
Initial_Positions= PSO.Swarm_Positions
Global_Best_Position = PSO.Global_Best_Position
Global_Best_Objectives = PSO.Global_Best_Objective
# In[]:
np.random.seed(seed=1)    
Obj2 = np.random.randint(-40,5,size = 10) 
Cons = np.array([np.random.randint(0,10,size = 10) for j in range(10)])
b = np.array([120, 100, 80, 80, 90,120, 110, 90, 90, 110]) 
c = 100
  
PSO2 = PSO_Algorithm(500, 300)         
Pso2 = PSO2.Fit_PSO(Obj2 , Cons, b, C = c, lower_bound = 0, upper_bound = 10, w = 0.65, c1 = 2, c2 = 2)
Positions2 = PSO2.Swarm_Best_Positions
Initial_Positions2= PSO2.Swarm_Positions
Global_Best_Position2 = PSO2.Global_Best_Position
Global_Best_Objectives2 = PSO2.Global_Best_Objective 
Violation =  np.dot(Cons, Global_Best_Position2) - b 
print(c*np.sum(Violation[Violation>0]))
