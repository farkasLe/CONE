
from time import time
import numpy as np
import pandas as pd
import copy
import math
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import itertools
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

Rand=np.random


metatrail=10
Totalbudget=1000
TDLMCsize=20


###region Algorithms
def YgridGenerator(TDLMCsize, TestProblem):
    if len(TestProblem.Y) == 1:
        return np.linspace(TestProblem.Y[0, 0], TestProblem.Y[0, 1], TDLMCsize)
    else:
        Ydimgrid = []
        YvaluePerDim=math.floor(math.pow(TDLMCsize, 1 / len(TestProblem.Y)))
        if YvaluePerDim==0:
            return False
        else:
            for d in range(len(TestProblem.Y)):
                Ydimgrid.append(np.linspace(TestProblem.Y[d, 0], TestProblem.Y[d, 1], YvaluePerDim))
            return np.array(itertools.product(*Ydimgrid))  # all y points on grid to do MC estimation on TDL
def ETDL(TestProblem, MC_Ys, surrogate):
    ETDL = len([y for y in MC_Ys if surrogate(TestProblem.CondOptimal(y), y) == min([surrogate(x, y) for x in TestProblem.X])])\
           / math.pow(math.floor(math.pow(TDLMCsize, 1 / len(TestProblem.Y))), len(TestProblem.Y))  # the record of TDL for one metatrail and one t
    return ETDL






def uniformsample(TestProblem):
    x=Rand.choice(TestProblem.X)
    y=Rand.uniform(TestProblem.Y[:, 0],TestProblem.Y[:, 1])
    return x,y

#region US-Krig
class USKrig:
    def __init__(self,TestProblem, l,totalbudget):
        self.testproblem=TestProblem
        self.l=l #it determines how many replications to do at each point
        self.T=totalbudget
        self.budgetleft=totalbudget
        self.GPR = []# a list of surrogate on each stage
        self.D={}#a list recording all observations
        self.Report=[]
        self.Ygrid=YgridGenerator(TDLMCsize,self.testproblem)
    def runalgo(self):
        print("USKrig: start initializing")
        inireport=self.initialize()# get a report of whether budget is enough after initilization
        if not inireport:
            print("Not enough budget for initilization")
        else:
            print("USKrig: Initialization done")
            while(self.budgetleft>0):
                print(f"USKrig: budget left {self.budgetleft}")
                x, y=uniformsample(self.testproblem) #uniformly sample a point on space
                BatchSize = min(self.budgetleft, self.l)  # take the minimum of budget left and a standard batch.
                BatchOut = []
                for i in range(BatchSize):
                    BatchOut.append(self.testproblem.simulation(x,y))
                localvar=np.var(BatchOut,ddof=1)
                localmean=np.mean(BatchOut)
                self.D[x].append([y,localmean,localvar])
                self.budgetleft-=BatchSize
                ##record the update surrogate
                Dict_Surrogate = self.surrogate()
                Surrogate = lambda x, y: Dict_Surrogate[x].predict(np.array([[y]]))
                self.Report.append([self.T - self.budgetleft, ETDL(self.testproblem,self.Ygrid,Surrogate)])
        print("Single trail USKrig done!")
        return self.Report
    def initialize(self):
        if self.budgetleft<self.l*len(self.testproblem.X):
            return False
        else:
            for x in self.testproblem.X:
                y=Rand.uniform(self.testproblem.Y[:, 0],self.testproblem.Y[:, 1])
                BatchOut=[]
                for i in range(self.l):
                    BatchOut.append(self.testproblem.simulation(x,y))
                localvar=np.var(BatchOut,ddof=1)
                localmean = np.mean(BatchOut)
                self.D[x]=[[y,localmean,localvar]]
                self.budgetleft-=self.l
            Dict_Surrogate=self.surrogate()
            Surrogate=lambda x,y: Dict_Surrogate[x].predict(np.array([[y]]))
            self.Report.append([self.T - self.budgetleft,ETDL(self.testproblem,self.Ygrid,Surrogate)])
            return True
    def surrogate(self):
        gpr = {}
        for x in self.testproblem.X:
            gpr[x] = GaussianProcessRegressor(alpha=np.array([est[2] for est in self.D[x]])).fit([est[0] for est in self.D[x]], [est[1] for est in self.D[x]])
        return gpr
#endregion

#region US-SNE
class USSNE:
    def __init__(self,TestProblem,totalbudget,Xi):
        self.testproblem=TestProblem
        self.T=totalbudget
        self.budgetleft=totalbudget
        self.c=0
        self.Xi=Xi
        self.D={}#a list recording all observations
        self.Report=[]
        if len(self.testproblem.Y)==1:
            self.c=(self.testproblem.Y[0,1]-self.testproblem.Y[0,0])/math.pow(2,-self.Xi)
        else:
            self.c=np.linalg.norm([self.testproblem.Y[d,1]-self.testproblem.Y[d,0]
                                   for d in range(len(self.testproblem.Y))])\
                   /math.pow(2,-self.Xi/len(self.testproblem.Y))
        self.Ygrid = YgridGenerator(TDLMCsize,self.testproblem)
        self.laststageguess= {}
    def runalgo(self):
        print("USSNE: start initializing")
        inireport=self.initialize()# get a report of whether budget is enough after initilization
        if not inireport:
            print("Not enough budget for initilization")
        else:
            while(self.budgetleft>0):
                print(f"USSNE: budget left {self.budgetleft}")
                x, y=uniformsample(self.testproblem) #uniformly sample a point on space
                self.D[x].append([y,self.testproblem.simulation(x,y)])
                self.budgetleft-=1
                ##record the update surrogate
                Surrogate = self.surrogate(self.D)
                self.Report.append([self.T - self.budgetleft, ETDL(self.testproblem, self.Ygrid, Surrogate)])
        print("Single trail USSNE done!")
        return self.Report
    def initialize(self):
        if self.budgetleft<2*len(self.testproblem.X):
            return False
        else:
            for x in self.testproblem.X:
                y1=Rand.uniform(self.testproblem.Y[:, 0],self.testproblem.Y[:, 1])
                y2 = Rand.uniform(self.testproblem.Y[:, 0], self.testproblem.Y[:, 1])
                self.D[x]=[[y1,self.testproblem.simulation(x,y1)],[y2,self.testproblem.simulation(x,y2)]]
                self.budgetleft-=2
            Surrogate = self.surrogate(self.D)
            self.Report.append([self.T - self.budgetleft, ETDL(self.testproblem, self.Ygrid, Surrogate)])
            print("USSNE: Initialization done")
            return True
    def surrogate(self,D):
        return lambda x,y: self.SNE(x,y,D)
    def SNE(self, x, y, D):
        list_m = [
            [math.floor(math.pow(np.linalg.norm(np.array(y - est[0])) / self.c, -len(self.testproblem.Y) / self.Xi)),
             est[1]] for est in D[x]]  # calculate the m value for each observations w.r.t. current y
        list_m.sort(key=lambda x: x[0])
        i = 1
        if (x,y) in self.laststageguess:
            i=self.laststageguess[(x,y)]# start from searching number of points within ball i
        sample = []
        while True:
            current_sample = [m[1] for m in list_m if m[0] >= i]
            if len(current_sample) < i:
                self.laststageguess[(x,y)]=i-1
                break
            sample = current_sample
            i += 1  # check i+1 in the next loop
        return [np.mean(sample), np.var(sample, ddof=1)]
#endregion

#region RES-SNE
class CONE:
    def __init__(self, TestProblem, totalbudget, Xi, upper=100, lower=0.01,
                 Y_weight=None, warmup=True, warmup_K=50):
        """
        :param TestProblem: Object defining the prescriptive problem
        :param totalbudget: Total simulation budget
        :param Xi: shrinking speed controller
        :param upper: initial M_U upper bound on sampling weight
        :param lower: initial M_L lower bound on sampling weight
        :param Y_weight: optional per-dimension distance weights for SNE (default None = raw distances)
        :param warmup: enable warm-up phase to calibrate M_L/M_U
        :param warmup_K: number of uniform samples during warm-up
        """
        self.testproblem = TestProblem
        self.T = totalbudget
        self.budgetleft = totalbudget
        self.M_U = upper
        self.M_L = lower
        self.Xi = Xi
        self.Y_weight = Y_weight
        self.warmup = warmup
        self.warmup_K = warmup_K
        self.D = {}  # observations: self.D[x] = [[y, sim_output], ...]
        self.Report = []
        self.Surrogate = None
        self.Ygrid = YgridGenerator(TDLMCsize, self.testproblem)

        # Compute c constant
        d = len(self.testproblem.Y)
        ranges = np.array([self.testproblem.Y[dim, 1] - self.testproblem.Y[dim, 0]
                           for dim in range(d)])
        if self.Y_weight is not None:
            ranges = ranges * self.Y_weight
        if d == 1:
            self.c = ranges[0] / math.pow(2, -self.Xi)
        else:
            self.c = np.linalg.norm(ranges) / math.pow(2, -self.Xi / d)

    def _uniformsample(self):
        x = self.testproblem.X[Rand.randint(0, len(self.testproblem.X))]
        y = Rand.uniform(self.testproblem.Y[:, 0], self.testproblem.Y[:, 1])
        return x, y

    def _compute_sampling_weight(self, x, y, sne):
        """
        Compute a(x,y) = u*(x,y)^{-(1+xi)} via the document's optimization.

        For the estimated-best xhat at y:
          u_0 in (0, min_x delta_x^2 / sigma_xhat^2)
          u_x = (delta_x^2 - sigma_xhat^2 * u_0) / sigma_x^2  for x != xhat
          Minimize h(u_0) = u_0^{-(1+xi)} + sum_{x!=xhat} u_x^{-(1+xi)}

        Returns a(x,y) clipped to [M_L, M_U].
        """
        # Evaluate SNE at all alternatives
        sne_vals = {}
        for xx in self.testproblem.X:
            sne_vals[xx] = sne(xx, y)  # [mean, variance]

        # Find estimated best xhat
        xhat = min(self.testproblem.X, key=lambda xx: sne_vals[xx][0])
        f_hat = sne_vals[xhat][0]
        var_hat = sne_vals[xhat][1]

        # Guard: if variance of best is zero or nan, return M_U
        if var_hat <= 0 or np.isnan(var_hat):
            return self.M_U

        # Compute delta_x^2 and sigma_x^2 for x != xhat
        others = [xx for xx in self.testproblem.X if xx != xhat]
        if len(others) == 0:
            return self.M_U

        delta_sq = {}  # delta_x^2 = (mean_x - mean_xhat)^2
        sigma_sq = {}  # sigma_x^2 = var_x
        for xx in others:
            delta_sq[xx] = (sne_vals[xx][0] - f_hat) ** 2
            sigma_sq[xx] = sne_vals[xx][1]
            if sigma_sq[xx] <= 0 or np.isnan(sigma_sq[xx]):
                sigma_sq[xx] = 1e-10  # small positive fallback

        # Upper bound for u_0: min_{x!=xhat} delta_x^2 / var_hat
        u0_upper = min(delta_sq[xx] / var_hat for xx in others)
        if u0_upper <= 1e-12:
            return self.M_U

        eps = 1e-10
        xi = self.Xi

        # Guard: if feasible interval is too small, skip optimization
        if u0_upper <= 2 * eps:
            return self.M_U

        def h(u0):
            # u_0 term
            val = np.power(u0, -(1 + xi))
            # sum over x != xhat
            for xx in others:
                u_x = (delta_sq[xx] - var_hat * u0) / sigma_sq[xx]
                if u_x <= 0:
                    return 1e30  # infeasible
                val += np.power(u_x, -(1 + xi))
            return val

        result = minimize_scalar(h, bounds=(eps, u0_upper - eps), method='bounded')

        if not result.success or np.isnan(result.fun):
            return self.M_U

        u0_star = result.x

        # Compute u*(x,y) for the queried x
        if x == xhat:
            u_star = u0_star
        else:
            u_star = (delta_sq[x] - var_hat * u0_star) / sigma_sq[x]
            if u_star <= 0:
                return self.M_U

        a_raw = np.power(u_star, -(1 + xi))

        # Clip to [M_L, M_U]
        return max(self.M_L, min(self.M_U, a_raw))

    def warmup_phase(self):
        """Run warmup_K uniform samples, compute a(x,y) for each,
        then set M_L = 10th percentile, M_U = 90th percentile."""
        a_values = []
        sne = self.surrogate(self.D)
        for _ in range(self.warmup_K):
            x, y = self._uniformsample()
            # Simulate and record
            self.D[x].append([y, self.testproblem.simulation(x, y)])
            self.budgetleft -= 1
            if self.budgetleft <= 0:
                break
            # Update surrogate and compute sampling weight
            sne = self.surrogate(self.D)
            a_val = self._compute_sampling_weight(x, y, sne)
            if np.isfinite(a_val):
                a_values.append(a_val)

        if len(a_values) >= 2:
            self.M_L = np.percentile(a_values, 10)
            self.M_U = max(self.M_L + 1e-6, np.percentile(a_values, 90))
        print(f"CONE warmup done: M_L={self.M_L:.4f}, M_U={self.M_U:.4f}")

    def runalgo(self):
        print("CONE: start initializing")
        inireport = self.initialize()
        if not inireport:
            print("Not enough budget for initialization")
            return self.Report

        # Optional warm-up phase
        if self.warmup and self.budgetleft > 0:
            self.warmup_phase()

        # Main loop: rejection sampling
        while self.budgetleft > 0:
            while True:
                x, y = self._uniformsample()
                sne = self.surrogate(self.D)
                a_xy = self._compute_sampling_weight(x, y, sne)
                # Accept with probability a(x,y) / M_U
                if Rand.uniform(0, 1) < a_xy / self.M_U:
                    break
            self.D[x].append([y, self.testproblem.simulation(x, y)])
            self.budgetleft -= 1
            print(f"CONE: budget left {self.budgetleft}")
            # Record ETDL if Ygrid is available
            if self.Ygrid is not None:
                Surrogate = self.surrogate(self.D)
                self.Report.append([self.T - self.budgetleft,
                                    ETDL(self.testproblem, self.Ygrid, Surrogate)])

        self.Surrogate = self.surrogate(self.D)
        print("Single trail CONE done!")
        return self.Report

    def final_xstar(self, y):
        """Return the estimated optimal decision for state y."""
        return min(self.testproblem.X, key=lambda x: self.Surrogate(x, y)[0])

    def initialize(self):
        if self.budgetleft < 2 * len(self.testproblem.X):
            return False
        for x in self.testproblem.X:
            y1 = Rand.uniform(self.testproblem.Y[:, 0], self.testproblem.Y[:, 1])
            y2 = Rand.uniform(self.testproblem.Y[:, 0], self.testproblem.Y[:, 1])
            self.D[x] = [[y1, self.testproblem.simulation(x, y1)],
                         [y2, self.testproblem.simulation(x, y2)]]
            self.budgetleft -= 2
        if self.Ygrid is not None:
            Surrogate = self.surrogate(self.D)
            self.Report.append([self.T - self.budgetleft,
                                ETDL(self.testproblem, self.Ygrid, Surrogate)])
        print("CONE: Initialization done")
        return True

    def surrogate(self, D):
        return lambda x, y: self.SNE(x, y, D)

    def SNE(self, x, y, D):
        d = len(self.testproblem.Y)
        exponent = -d / self.Xi
        list_m = []
        for est in D[x]:
            if self.Y_weight is not None:
                dist = np.linalg.norm(np.array(y - est[0]) * self.Y_weight)
            else:
                dist = np.linalg.norm(np.array(y - est[0]))
            ratio = dist / self.c
            if ratio == 0:
                m_val = 10**9  # treat zero-distance as very large m
            else:
                m_val = math.floor(math.pow(ratio, exponent))
            list_m.append([m_val, est[1]])

        list_m.sort(key=lambda x: x[0])
        i = 1
        sample = []
        while True:
            current_sample = [m[1] for m in list_m if m[0] >= i]
            if len(current_sample) < i:
                break
            sample = current_sample
            i += 1

        # Fallback: if Psi < 2, use 2 nearest neighbors
        if i < 2:
            if self.Y_weight is not None:
                list_dist = [[np.linalg.norm(np.array(y - est[0]) * self.Y_weight),
                              est[1]] for est in D[x]]
            else:
                list_dist = [[np.linalg.norm(np.array(y - est[0])),
                              est[1]] for est in D[x]]
            list_dist.sort(key=lambda x: x[0])
            sample = [jj[1] for jj in list_dist[:2]]

        return [np.mean(sample), np.var(sample, ddof=1)]
#endregion
###endregion


###regionTestProblem
class TestProblem:
    def __init__(self, X, Y, f, sigma):
        self.X=X
        self.Y=Y
        self.f=f
        self.sigma=sigma
    def simulation(self, x, y):
        return self.f(x, y) + Rand.normal(scale=self.sigma(x, y))
    def CondOptimal(self,y):
        Opt=min([self.f(x,y) for x in self.X])
        for x in self.X:
            if self.f(x,y)==Opt:
                return x
##region 1d function
def F1(x,y):
    if isinstance(y, np.ndarray):
        y = y[0]
    if x==1:
        return 10/(y+1)**2*math.sin((math.e)**(y+1))
    elif x==2:
        return 0
    elif x==3:
        return -10 / (y + 0.8) ** 2 * math.sin((math.e) ** (y + 0.8))
    else:
        print("wrong x")

def Sigma1(x,y):
    if isinstance(y, np.ndarray):
        y=y[0]
    if x==1:
        return 0.5*(math.sin(16*y)+1.2)
    elif x==2:
        return 0.5*(math.sin(8*y)+1.2)
    elif x==3:
        return 0.5*(math.sin(4*y)+1.2)
    else:
        print("wrong x")

F1d=TestProblem([1,2,3],np.array([[0,2]]),F1,Sigma1)
##endregion
##region 2d function
##endregion
##region real problem function
##endregion
###endregion


###Output show

##region Test


# Set up the mean and variance data
methods = ['USKrig', 'USSNE', 'CONE']


if __name__ == "__main__":
    Rand.seed(0)
    """#USKrig:
    #[0.095, 0.007725]
    #USSNE:
    #[0.5450000000000002, 0.04272499999999999]
    #CONE:
    #[0.705, 0.005725000000000001]
    mean = [0.095,0.5450000000000002,0.705]#[0.0, 0.32, 0.6]
    variance = [0.007725,0.04272499999999999,0.005725000000000001]#[0.0, 0.0376, 0.007999999999999997]

    # Convert variance to standard deviation
    std_dev = np.sqrt(variance)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(7,5),dpi=250)

    # The x locations for the groups
    x_pos = np.arange(len(methods))

    # Plot bars
    ax.bar(x_pos, mean, yerr=std_dev, align='center', alpha=0.5, ecolor='black', capsize=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Methods')
    ax.set_ylabel('1-TDL')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Show the figure
    plt.tight_layout()
    plt.show()
    ##endregion

    ##region Test 2
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta

    start = datetime.strptime("08:00", "%H:%M")
    end = datetime.strptime("12:00", "%H:%M")
    delta = timedelta(minutes=30)
    times = mdates.drange(start, end, delta)

    fig, ax = plt.subplots()
    ax.plot_date(times, [0]*len(times), '-')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation='vertical')
    plt.show()"""

##endregion



















    plt.figure(figsize=(7,5),dpi=250)
    ##region Run USKrig
    ML_ETDL_USKrig=[]#the record of TDL across metatrail and stage t

    for i in range(metatrail):# run each metatrail
        print('%d USKrig'%(i+1))
        L_ETDL_USKrig=[]#the record of TDL for one metatrail and across t
        Report_USKrig=USKrig(F1d, 5, Totalbudget).runalgo()
        ML_ETDL_USKrig.append(Report_USKrig)
        #plot estimated TDL against budget spent
    plt.plot(np.mean(ML_ETDL_USKrig, axis=0)[:, 0],np.mean(ML_ETDL_USKrig, axis=0)[:, 1],'-k',label='USKrig')
    print("USKrig meta run done!")
    ##endregion

    ##region Run USSNE
    ML_ETDL_USSNE=[]#the record of TDL across metatrail and stage t

    for i in range(metatrail):# run each metatrail
        print('%d USSNE' % (i + 1))
        L_ETDL_USSNE=[]#the record of TDL for one metatrail and across t
        c=0
        Report_USSNE=USSNE(F1d,Totalbudget,0.5).runalgo()
        ML_ETDL_USSNE.append(Report_USSNE)
        #plot estimated TDL against budget spent
    plt.plot(np.mean(ML_ETDL_USSNE, axis=0)[:, 0],np.mean(ML_ETDL_USSNE, axis=0)[:, 1],'-b',label='USSNE')
    print("USSNE meta run done!")
    ##endregion

    ##region Run CONE
    ML_ETDL_CONE=[]#the record of TDL across metatrail and stage t

    for i in range(metatrail):# run each metatrail
        print('%d CONE' % (i + 1))
        L_ETDL_CONE=[]#the record of TDL for one metatrail and across t
        c=0
        experiment=CONE(F1d,Totalbudget,Xi=1,upper=30,lower=1)
        Report_CONE=experiment.runalgo()
        ML_ETDL_CONE.append(Report_CONE)
        #plot estimated TDL against budget spent
    plt.plot(np.mean(ML_ETDL_CONE, axis=0)[:, 0],np.mean(ML_ETDL_CONE, axis=0)[:, 1],'-r',label='CONE')
        # Show the plot



    print("CONE meta run done!")
    ##endregion


    plt.title('Total decision loss against #Total observations')
    plt.xlabel('Total observations')
    plt.ylabel('1-Total decision loss')
    plt.legend()
    plt.savefig('TDLvObs.png')

    if metatrail == 1:
        data = [[a[0][0] for a in experiment.D[i]] for i in experiment.D]
        # Plot the histogram
        colour=['r','b','g']
        for i in range(len(data)):
            print(data[i])
            plt.hist(data[i],bins=100, color=colour[i])  # Specify the bins and edge color
            plt.title(f'Histogram of y for f_{i+1}')
            plt.xlabel('y-location')
            plt.ylabel('frequency')
            plt.savefig(f'hist_y_{i+1}.png')