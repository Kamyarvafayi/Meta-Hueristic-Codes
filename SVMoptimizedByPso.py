import numpy as np
class SVM_optimized_by_PSO:
    def __init__(self, Iteration = 10000, Swarm_Num = 10):
        self.Iteration = Iteration
        self.Swarm_Num = Swarm_Num
    def Initial_Positions(self):
        self.Initial_Position = np.array([np.random.uniform(self.Low, self.High, size = self.Input.shape[0]) for i in range(self.Swarm_Num)])
        return self.Initial_Position.copy()
    def Find_Objective(self, New_Solution):
        Objective = 0
        for i in range(self.Input.shape[0]):
            for j in range(self.Input.shape[0]):
                Objective += New_Solution[i]*self.Target[i]*self.Kernel[i,j]*self.Target[j]*New_Solution[j]
        print(0.5*Objective - np.sum(New_Solution)+ self.C*np.abs(New_Solution.dot(self.Target)))
        return 0.5*Objective - np.sum(New_Solution) + self.C*np.abs(New_Solution.dot(self.Target))
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
    def Linear_Kernel(self):
        self.Kernel = np.array([[self.Input[i].dot(self.Input[j]) for i in range(self.Input.shape[0])] for j in range(self.Input.shape[0])])
            
    def Find_Global(self):
        self.Global_Best_Position = self.Swarm_Best_Positions[np.argmin(self.Swarm_Best_Objectives)]
        self.Global_Best_Objective = np.min(self.Swarm_Best_Objectives)
        print(self.Global_Best_Position,": ", self.Global_Best_Objective, "\n")
    def Fit_SVM(self, Input, Target,   C = 1, lower_bound = 0, Lambda = 1, w = 1, c1=2, c2=2):
        self.Input = Input
        self.Target = Target
        self.C = C
        self.Low = lower_bound
        #self.High = 1/(Lambda*2*self.Input.shape[0])
        self.High = Lambda
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.Linear_Kernel()
        self.Swarm_Positions = self.Initial_Positions().copy()
        self.Swarm_Best_Positions = self.Swarm_Positions 
        self.Swarm_Best_Objectives = self.Swarm_Objectives = np.array([self.Find_Objective(self.Swarm_Positions[i]) for i in range(self.Swarm_Num)])
        self.Global_Best_Position = self.Swarm_Best_Positions[np.argmin(self.Swarm_Best_Objectives)]
        self.Global_Best_Objective = np.min(self.Swarm_Best_Objectives)
        self.V = [np.random.uniform(-(self.High-self.Low)*0.1,(self.High-self.Low)*0.1, size = self.Input.shape[0]) for i in range(self.Swarm_Num)]
        #self.V = [np.ones(shape=[1,self.Input.shape[0]])*(self.High-self.Low)*0.1 for i in range(self.Swarm_Num)]
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
from sklearn import datasets
Input , Target = datasets.make_moons(n_samples=1000, shuffle=True, noise=0)
Target[Target==0] = -1
# In[]:
SVM = SVM_optimized_by_PSO(Iteration = 30, Swarm_Num = 5)
Lambda = 1/(2*Input.shape[0])
SVM.Fit_SVM(Input, Target, Lambda = 1, C= 100)
solution = SVM.Global_Best_Position
solution2 = solution.copy()
#solution2 = solution2.reshape(solution2.shape[0],1)
solution2[solution==1] = 0
#Target = Target.reshape(Target.shape[0],1)
cy = solution2[solution2>0]*Target[solution2>0]
x = Input[solution2>0,:]
Weight = np.array([[0.0],[0.0]])
for i in range(solution2[solution2>0].shape[0]):
    for j in range(2):
        Weight[j] += cy[i]*x[i,j]

    
# In[]:
from sklearn import svm
clf = svm.SVC(C = 100, kernel='linear')
clf.fit(Input, Target)
print(clf.coef_)
