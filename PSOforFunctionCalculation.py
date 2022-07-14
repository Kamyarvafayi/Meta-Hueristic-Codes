import numpy as np
class PSO_Algorithm:
    def __init__(self, Iteration = 10000, Swarm_Num = 10):
        self.Iteration = Iteration
        self.Swarm_Num = Swarm_Num
    def Initial_Positions(self):
        self.Initial_Position = np.array([np.random.uniform(self.Low, self.High, size = 1) for i in range(self.Swarm_Num)])
        return self.Initial_Position.copy()
    def Find_Objective(self, New_Solution):
        Objective = np.exp(New_Solution*-(0.1))*np.sin(2*np.pi*New_Solution)*np.sin(0.1*np.pi*New_Solution)
        print(New_Solution,": ", Objective, "\n\n\n")
        return Objective
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
    def Fit_PSO(self, lower_bound = 0, upper_bound = 1, w = 0.6, c1=2, c2=2):
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
        self.V = [np.random.uniform(-1,1, size = 1) for i in range(self.Swarm_Num)]
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
PSO = PSO_Algorithm(500, 50)           
Pso = PSO.Fit_PSO(lower_bound = -5, upper_bound = 40, w = 1)
Positions = PSO.Swarm_Best_Positions
Initial_Positions= PSO.Swarm_Positions
Global_Best_Position = PSO.Global_Best_Position
Global_Best_Objectives = PSO.Global_Best_Objective

# In[]: 
import matplotlib.pyplot as plt
x = np.linspace(-5,40, 10000)
Objective = np.exp(x*-(0.1))*np.sin(2*np.pi*x)*np.sin(0.1*np.pi*x)
plt.plot(x, Objective)
plt.show()
