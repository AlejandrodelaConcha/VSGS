# -----------------------------------------------------------------------------------------------------------------
# Title:  variable_selection_detector
# Author(s):      
# Initial version:  [insert date in 20200530]
# Last modified:    [insert date in 20200530]
# Designed for:     Python 3.7.0, macOS Catalina version 10.15.1
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This algorithm is an implementation of the algorithm presented in
#  our paper
# -----------------------------------------------------------------------------------------------------------------
# Notes:
# -----------------------------------------------------------------------------------------------------------------
# Required libraries : ruptures, sklearn, numpy, scipy 
# -----------------------------------------------------------------------------------------------------------------
# Key words:change-point, Kernel methods, model selection 
# -----------------------------------------------------------------------------------------------------------------


import ruptures as rpt 
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.special import comb
import scipy
import copy
from pygsp import graphs, filters, plotting
import matplotlib.pyplot as plt

class VSGS():
    
  
    def __init__(self,D_max,U,PSD,coefs=None):
        
        ##### Function initializing the class
        
        ### Input
        
        ## D_max = maximum number of change-points.
        ## coefs= coefficients related with the penalization terms. 
        ## U eigenvectors of the GSO
        ## PSD= Power Spectral Density. 
        

        self.D_max=D_max
        self.coefs=coefs
        self.U=U
        self.PSD=PSD
        
        
        
    def fit(self,data,Lambda):
        
        #### Function diferent partitions from size 0 up to size D_max for all the values in the grid of lambda values.
        
        ### Input 
        
        ## data matrix of size Txp where T is the time horizon, p the number of covariance 
        ## Lambda: values of the lambda parameter to be tested. 
        
        
        ######## Initializing variables 
        self.T=data.shape[0]
        self.p=data.shape[1]
        self.change_points_grid=[]
        self.relevant_frecuencies=[]
        self.Dm=[]
        self.Lambda=[]
        self.partitions_costs=[]
        
        Lambda.sort()
        
        ######## Translate the problem to the Spectral domain 
        self.fourier_signal= data.dot(self.U)
        self.fourier_signal=self.fourier_signal/np.sqrt(self.PSD) #### Changed. 
        mean_fourier=np.apply_along_axis(np.mean,0,self.fourier_signal)
        
        
        ############# Solve the change-point detection for each of the levels of sparsity. 
        
        aux_relevant_frecuencies=np.where(np.abs(mean_fourier)>Lambda[0]/2.)[0]
        self.relevant_frecuencies.append(aux_relevant_frecuencies)
        self.Lambda.append(Lambda[0])
        self.Dm.append(len(aux_relevant_frecuencies))
        reduced_signal=self.fourier_signal[:,aux_relevant_frecuencies]
        linear_cost = rpt.costs.CostMl(metric=np.eye(reduced_signal.shape[1])).fit(reduced_signal)
        aux_change_points=self.fit_lambda(reduced_signal,linear_cost)
        self.partitions_costs.append(copy.deepcopy(self.get_partitions_cost_lamb(aux_change_points,linear_cost,aux_relevant_frecuencies)))
        self.change_points_grid.append(copy.deepcopy(aux_change_points))
        
       
        for i in range(1,len(Lambda)):
      
            
            ####### First Lasso to select relevant frecuencies
            
            aux_relevant_frecuencies=np.where(np.abs(mean_fourier)>Lambda[i]/2.)[0]
           
         
            
            ####### Finding the sequence of change-points for each value in the grid. 
       
            if (len(aux_relevant_frecuencies) < self.Dm[len(self.Dm)-1] and len(aux_relevant_frecuencies)>0):
                 self.relevant_frecuencies.append(aux_relevant_frecuencies)
                 self.Lambda.append(Lambda[i])
                 self.Dm.append(len(aux_relevant_frecuencies))
                 reduced_signal=self.fourier_signal[:,aux_relevant_frecuencies]
                 linear_cost = rpt.costs.CostMl(metric=np.eye(reduced_signal.shape[1])).fit(reduced_signal)
                 aux_change_points=self.fit_lambda(reduced_signal,linear_cost)
                 self.partitions_costs.append(copy.deepcopy(self.get_partitions_cost_lamb(aux_change_points,linear_cost,aux_relevant_frecuencies)))
                 self.change_points_grid.append(copy.deepcopy(aux_change_points))
                
                 
                
            
    
    def fit_lambda(self,data,cost_function):
        
        #### Function finding the diferent partitions from size 0 up to size D_max given a value of lambda. 
        
        ### Input 
        ## data= matrix of size Txm where T is the time horizon, m the number of relevant covariances.
        ## cost_function=cost function object from the ruptures library
        
        change_points=[] 
        change_points.append([self.T])
     
        detector = rpt.Dynp(custom_cost=cost_function,min_size=2,jump=2).fit(data)

        for d in range(1,self.D_max):
            change_points.append(detector.predict(n_bkps=d))
            
        return change_points
            
    def predict(self,lower_bound_tau=None):
        ### Function selecting the optimal number and the amount of sparcity of change-points via a model selection approach
        
        ### Input 
        ## lower_bound_tau: minimal_number_of change_points
        
        ### Output 
        ## self.change_points
        
        term_Dm=[]
        
        self.get_constants(lower_bound_tau)
        
        partition_cost=[]
        
        for l in range(len(self.Dm)):
            partition_cost.extend(self.partitions_costs[l])
            term_Dm.extend(self.D_max*[self.Dm[l]])
            
         
        term_Dm=np.array(term_Dm).flatten()
        rank_D=list(range(1,self.D_max+1))
        
        linear_term_D_tau=np.array(len(self.Dm)*rank_D).flatten()
        log_term_D_tau=np.log(self.T/linear_term_D_tau)*linear_term_D_tau
        
        
        penalized_cost_function=np.array(partition_cost)+(1./self.T)*(self.coefs[0]*term_Dm+
                                                                      self.coefs[1]*linear_term_D_tau+
                                                                      self.coefs[2]*log_term_D_tau)
           
        
        index=np.argmin(penalized_cost_function)
       
        index_change_points=np.where(rank_D==linear_term_D_tau[index])[0][0]#### Changed 
        index_lambda=np.where(self.Dm==term_Dm[index])[0][0]
        self.change_points=self.change_points_grid[index_lambda][index_change_points]     
        
        ##### Selecting best amount of sparsity 
                 
        self.optimal_lamb=self.Lambda[index_lambda]
        self.get_mu()
        
        change_points=copy.deepcopy(self.change_points)
        mu=copy.deepcopy(self.mu)
                        
        return change_points,mu
    
    def get_mu(self):
        ### Function estimating the mean for the Stream of Graph Signals in each of found segments.
               
        self.mu=[]
             
        self.cost_L1=0
         
        lower_bound=0
        upper_bound=self.change_points[0]
        mean_fourier=np.apply_along_axis(np.mean,0,self.fourier_signal[lower_bound:upper_bound])
        non_relevant_frecuencies=np.where(np.abs(mean_fourier)<(self.optimal_lamb/2.))[0]
        mean_lasso=np.sign(mean_fourier)*(np.abs(mean_fourier)-(self.optimal_lamb/2.))
        if len(non_relevant_frecuencies)>0:
            mean_fourier[non_relevant_frecuencies]=np.zeros(len(non_relevant_frecuencies))
            mean_lasso[non_relevant_frecuencies]=0
                
        self.cost_L1+=(np.sum((mean_lasso-np.array(self.fourier_signal[lower_bound:upper_bound]))**2)+self.optimal_lamb*np.sum(np.abs(mean_lasso))*(upper_bound-lower_bound))/self.T
        self.mu.append(mean_fourier*np.sqrt(self.PSD))
        
        
        for d in range(1,len(self.change_points)):
            lower_bound=self.change_points[d-1]
            upper_bound=self.change_points[d]
            mean_fourier=np.apply_along_axis(np.mean,0,self.fourier_signal[lower_bound:upper_bound])
            non_relevant_frecuencies=np.where(np.abs(mean_fourier)<(self.optimal_lamb/2.))[0]
               
            mean_lasso=np.sign(mean_fourier)*(np.abs(mean_fourier)-(self.optimal_lamb/2.))
            if len(non_relevant_frecuencies)>0:
                mean_lasso[non_relevant_frecuencies]=0
                mean_fourier[non_relevant_frecuencies]=0
            self.cost_L1+=(np.sum((mean_lasso-np.array(self.fourier_signal[lower_bound:upper_bound]))**2)+self.optimal_lamb*np.sum(np.abs(mean_lasso))*(upper_bound-lower_bound))/self.T
            self.mu.append(mean_fourier*np.sqrt(self.PSD))
                     
    def get_partitions_cost_lamb(self,change_points,cost_function,relevant_frecuencies):
        #### Function estimating the Min Squares error cost for a given value of lambda parameter. 
        
        ### Input:
        ## change_points= grid of change-points for 
        ## cost_function=cost function object from the ruptures library
        ## relevant_frquencies= index of the frecuencies to kept.
        
        partition_cost=np.zeros(self.D_max)
        
        
        non_relevant_frecuencies=list(set(range(self.p))-set(relevant_frecuencies))
        
        lower_t=0
        upper_t=change_points[0][0]
        partition_cost[0]=(1./self.T)*(cost_function.error(lower_t,upper_t)+np.sum(np.asarray(self.fourier_signal[lower_t:upper_t,non_relevant_frecuencies])**2))
        

        for d in range(1,len(change_points)):
            
          
            lower_t=0
            upper_t=change_points[d][0]
            partition_cost[d]=(1./self.T)*(cost_function.error(lower_t,upper_t)+np.sum(np.asarray(self.fourier_signal[lower_t:upper_t,non_relevant_frecuencies])**2))
            
            for i in range(1,len(change_points[d])):
                
                lower_t=change_points[d][i-1]
                upper_t=change_points[d][i]
                
                partition_cost[d]+=(1./self.T)*(cost_function.error(lower_t,upper_t)+np.sum(np.asarray(self.fourier_signal[lower_t:upper_t,non_relevant_frecuencies])**2))
            
        return partition_cost
    
    def get_constants(self,lower_bound_tau=None): 
        
        #### Function computing the constants which are required for the computation of the 
        ## optimal number of change-points via the slope heuristic.
        
        ### Input 
        ## lower_bound_tau: minimal_number_of change_points
        
        y=[]
        term_Dm=[]
        
        if lower_bound_tau is None:
            lower_bound_tau=int(np.floor(0.6*self.D_max))
            lower_bound_m=int(np.floor(0.6*self.p))
            
        rank_D=list(range(lower_bound_tau,self.D_max+1))
        
        aux_Dm=[x for x in self.Dm if x>=lower_bound_m]
        
        for l in range(len(aux_Dm)):
            index=[int(x)-1 for x in rank_D]
            y.extend(self.partitions_costs[l][index])
            term_Dm.extend(len(rank_D)*[aux_Dm[l]])
                      
        y=np.array(y)

        dimension_m=np.array(term_Dm).flatten()
        linear_term_D_tau=np.array(len(aux_Dm)*rank_D).flatten()
        log_term_D_tau=np.log(self.T/linear_term_D_tau)*linear_term_D_tau
        x=np.vstack((dimension_m,linear_term_D_tau,log_term_D_tau)).transpose()/self.T
        self.coefs=LinearRegression().fit(x,y).coef_
        self.coefs=-2.*self.coefs



    
    
def estimate_PSD(data,G,U,lamb,method="perraudin",plot=True):  
    
    #### This functions estimates the PSD of the graph signal with two different methods.
    ## The maximum-likelihood estimator (likelihood)
    ## and the method described in "Stationary signal processing on graphs" (Perraudin 2017)
    
    ### Input.
    ## data= matrix of size Txp where T is the time horizon, p the number of covariance        
    ## G= graph over the which the signal is defined 
    ## U= eigenvectors of the GSO
    ## lamb= eigenvalues of the GSO
    ## method = which algorithm to use in order to estimate the GFT
    ## plot = whether or not plot the PSD estimator
    
    ### Output
    ## PSD= Power Spectral Density
    
       
        p=data.shape[1]
        N=data.shape[0]
        
        if method=="likelihood":
            
            
            PSD=np.diag(np.cov(U.transpose().dot(data.transpose())))
            if plot:
                M=300 #### Number of filters 
                m=np.arange(0,M)
                l_max=np.max(lamb)           
                tau=((M+1)*l_max)/M**2
                plt.plot(m*tau,PSD)
                plt.show()
            
            
            return(PSD)
            
        
        if method=="perraudin":
          
            ###### Parameters initialization
            
            
            M=100 #### Number of filters
            degree=15
            m=np.arange(0,M)
            l_max=np.max(lamb)
            noise=np.random.normal(size=(p,10))
            norm_filters=np.zeros(M)
            norm_localized_filters=np.zeros(M)
            PSD=np.zeros(M)
            tau=((M+1)*l_max)/M**2
            
           ##### Applying filters to noise 
            for i in m:
                G_filter=filters.Filter(G, lambda x: gaussian_filter(x,i,M,l_max))
                filter_noise=G_filter.filter(noise)
                norm_filters[i]=np.sum(np.apply_along_axis(lambda x: np.mean(x**2),1,filter_noise))
                localized_filter=G_filter.filter(data.transpose())
                norm_localized_filters[i]=np.sum(np.apply_along_axis(lambda x: np.mean(x**2),1,localized_filter))
                PSD[i]=norm_localized_filters[i]/norm_filters[i]
            
            PSD=PSD
            coeff=np.polyfit(m*tau,PSD,deg=degree)
            if plot:
                plt.plot(m*tau,PSD)
                plt.show()
                
            p = np.poly1d(coeff)
            PSD=p(lamb)
            index_PSD=np.where(PSD<=0)[0]
            PSD[index_PSD]=np.min(PSD[PSD>0])
            
            return(PSD)      
    
    
def gaussian_filter(x,m,M,l_max):
    
    ### This function implement a Guassian Graph filter.
    
    ### Input
    ## x= point to be evaluated
    ## m = numberof the current filter
    ## M = total number of filters
    ## l_max= maximum eigenvalue of the GSO
    
        sigma_2=((M+1)*l_max)/M**2
        tau=sigma_2
        return np.exp(-((x-m*tau)**2)/sigma_2)    
    