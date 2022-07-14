import numpy as np
# In[]:
class ICA_Class():
    def __init__(self, Iteration, Imperialist_Num, Colony_num):
        self.Max_Iteration = Iteration
        self.Imperialist_num = Imperialist_Num
        self.Colony_Num = Colony_num
    def Initialization(self):
        self.Initial_Positions = []
        for i in range(self.Imperialist_num):
            Positions = dict()
            Positions["Colony"] = np.array([np.random.uniform(self.Low, self.High, size = self.Obj_Weights.shape[0]) for i in range(self.Colony_Num)])
            Positions["Imperialist"] = np.random.uniform(self.Low, self.High, size = self.Obj_Weights.shape[0])
            self.Initial_Positions.append(Positions)
        return self.Initial_Positions.copy()
    def Revolution(self, Sample, Mu_Coef):
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
    def Assimilation(self, Imperialist, Colony):
        result = Colony + self.Beta*np.random.rand()*(Imperialist-Colony)
        result[result>self.High] = self.High
        result[result<self.Low] = self.Low
        return result
    def Find_Objective(self, New_Solution):
        Objective = np.dot(self.Obj_Weights, New_Solution)
        Constriant = np.array([np.dot(self.Const_Weights[i], New_Solution) for i in range(self.Const_Weights.shape[0])])
        Violation = Constriant-self.b
        #print(self.C*np.sum(Violation[Violation>0]),"\n\n\n")
        #print(Objective,"\n\n\n")
        return (Objective + self.C*np.sum(Violation[Violation>0]))
    def Empire_Cost(self, index):
        Imperialist_Cost = self.Find_Objective(self.Imperialist_Colonies[index]["Imperialist"])
        Colonies_Cost = np.average(np.array([self.Find_Objective(self.Imperialist_Colonies[index]["Colony"][i]) for i in range(self.Imperialist_Colonies[index]["Colony"].shape[0])]))
        Empire_Cost = Imperialist_Cost + Colonies_Cost*self.zeta
        return Empire_Cost
    def Exchange_Imp_Colony(self, Imperialist_Colony):
        ArgMin_Colony = np.argmin([self.Find_Objective(Imperialist_Colony["Colony"][i]) for i in range( Imperialist_Colony["Colony"].shape[0])])
        #print(ArgMin_Colony)
        Min_Colony = np.min([self.Find_Objective(Imperialist_Colony["Colony"][i]) for i in range(Imperialist_Colony["Colony"].shape[0])])
        if self.Find_Objective(Imperialist_Colony["Imperialist"]) > Min_Colony:
            temp = Imperialist_Colony["Imperialist"].copy()
            Imperialist_Colony["Imperialist"] = Imperialist_Colony["Colony"][ArgMin_Colony].copy()
            Imperialist_Colony["Colony"][ArgMin_Colony] = temp
        return Imperialist_Colony.copy()
    def Change_Colony_Imperialist(self, Imperialist_Index):
        random1 = np.random.randint(0, self.Imperialist_Colonies[Imperialist_Index]["Colony"].shape[0])
        random2 = Imperialist_Index
        while random2 == Imperialist_Index:
            random2 = np.random.randint(0,len(self.Imperialist_Colonies))
        self.Imperialist_Colonies[random2]["Colony"] = np.append(self.Imperialist_Colonies[random2]["Colony"], np.array([self.Imperialist_Colonies[Imperialist_Index]["Colony"][random1]]), axis =0)
        self.Imperialist_Colonies[Imperialist_Index]["Colony"] = np.delete(self.Imperialist_Colonies[Imperialist_Index]["Colony"], random1, axis = 0)       
    def Eliminate_IMPERIALIST(self):
        New_Imperialist_Colonies = self.Imperialist_Colonies.copy()
        for i in range(len(self.Imperialist_Colonies)):
            if self.Imperialist_Colonies[i]["Colony"].shape[0] == 0:
                Random = np.random.randint(0,len(self.Imperialist_Colonies)-1)
                temp = self.Imperialist_Colonies[i]["Imperialist"]
                New_Imperialist_Colonies.pop(i)
                New_Imperialist_Colonies[Random]["Colony"] = np.append(New_Imperialist_Colonies[Random]["Colony"], np.array([temp]), axis = 0) 
        self.Imperialist_Colonies = New_Imperialist_Colonies
    def Fit_ICA(self, Objective_Function, Constraint = np.array([0]), b = np.array([0]), C = 100, lower_bound = 0, upper_bound = 10, Rev_Pr = 0.1, Assimilation_Beta = 2, teta = np.pi/4, zeta = 0.3):
        self.Obj_Weights = Objective_Function
        self.Final_Objectives = []
        self.Low = lower_bound
        self.High = upper_bound  
        self.b = b
        self.C = C
        self.Const_Weights = Constraint
        self.Rev_Pr = Rev_Pr
        self.Beta = Assimilation_Beta
        self.teta = teta
        self.zeta = zeta
        # Initialization
        self.Imperialist_Colonies = self.Initialization()
        for k in range(len(self.Imperialist_Colonies)):
            self.Imperialist_Colonies[k] = self.Exchange_Imp_Colony(self.Imperialist_Colonies[k].copy())
        for Iter in range(self.Max_Iteration):
            # Assimilation
            for Imper in range(len(self.Imperialist_Colonies)):
                for Colony in range(self.Imperialist_Colonies[Imper]["Colony"].shape[0]):
                    self.Imperialist_Colonies[Imper]["Colony"][Colony] = self.Assimilation(self.Imperialist_Colonies[Imper]["Imperialist"], self.Imperialist_Colonies[Imper]["Colony"][Colony])
            # Revolution
            for Imper in range(len(self.Imperialist_Colonies)):
                for Colony in range(self.Imperialist_Colonies[Imper]["Colony"].shape[0]):
                    Random = np.random.rand()
                    if Random <= Rev_Pr: 
                        self.Imperialist_Colonies[Imper]["Colony"][Colony] = self.Revolution(self.Imperialist_Colonies[Imper]["Colony"][Colony], self.Obj_Weights.shape[0]/4)
            # Changing the position of best colony with the imperialist
            for k in range(len(self.Imperialist_Colonies)):
                self.Imperialist_Colonies[k] = self.Exchange_Imp_Colony(self.Imperialist_Colonies[k].copy())
            # Emperors' Cost
            self.Average_Cost = []
            for i in range(len(self.Imperialist_Colonies)):
                self.Average_Cost.append(self.Empire_Cost(i))
            # Imperialist Competition
            Weakest_Imp_Index = np.argmax(self.Average_Cost)
            self.Change_Colony_Imperialist (Weakest_Imp_Index)
            self.Eliminate_IMPERIALIST()
            # Changing the position of best colony with the imperialist
            for k in range(len(self.Imperialist_Colonies)):
                self.Imperialist_Colonies[k] = self.Exchange_Imp_Colony(self.Imperialist_Colonies[k].copy())
            # Final Objective Function
            Best_Costs = np.array([self.Find_Objective(self.Imperialist_Colonies[i]["Imperialist"]) for i in range(len(self.Imperialist_Colonies))])
            self.Final_Objectives.append(np.min(Best_Costs))
            print("Objective Function in step {} is:".format(Iter+1), self.Final_Objectives[-1])
# In[]:
np.random.seed(seed = 1)
Obj = np.random.randint(-40,5,size = 50)             
ICA = ICA_Class(200, 5, 30)           
Ica = ICA.Fit_ICA(Obj , lower_bound = 0, upper_bound = 10)
Initial_Positions= ICA.Initial_Positions
Imperialist_Colony = ICA.Imperialist_Colonies
Objective = ICA.Final_Objectives

# In[]: Constraint Problem

Cons = np.array([np.random.randint(0,5,size = 10) for j in range(10)])
#b = np.array([100, 100, 70, 70, 70,100, 50, 80, 90, 110]) 
b = np.array([30 for i in range(10)])
c = 1000 
Obj2 = np.random.randint(-40,5,size = 10) 
np.random.seed(seed = 1)             
ICA2 = ICA_Class(500, 10, 50)           
Ica2 = ICA2.Fit_ICA(Obj2 , lower_bound = 0, upper_bound = 10, Constraint=Cons, C=c, b=b, Rev_Pr = 0.5)
Initial_Positions2= ICA2.Initial_Positions
Imperialist_Colony2 = ICA2.Imperialist_Colonies
Objective2 = ICA2.Final_Objectives
Violation =  np.dot(Cons, Imperialist_Colony2[5]["Imperialist"]) - b            
