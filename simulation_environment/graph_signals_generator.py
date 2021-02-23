# -----------------------------------------------------------------------------------------------------------------
# Title:  graph_signals_generator
# Author(s):     
# Initial version:  [insert date in 20200319]
# Last modified:    [insert date in 20200320]
# Designed for:     Python 3.7.0, macOS Catalina version 10.15.1
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This algorithm generates simulations of stationary graph signals .
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# Required libraries : numpy,  pygsp 
# -----------------------------------------------------------------------------------------------------------------
# Key words:Graph signal processing. 
# -----------------------------------------------------------------------------------------------------------------

import numpy as np
from pygsp import graphs, filters, plotting


class change_point_generator():
    def __init__(self,generators,change_points):
        
        #### This function generates the stream of stationary graph signals to be segmented. 
        
        ## Input: 
        ## generators= list of stationary_graph_signal_simulator objects 
        ## change_points =list of change_point locations
        
        self.generators=generators
        self.change_points=change_points
        self.history=[]
       
    def generate_signal(self,**kargs):
        
        #### Generates the stream of graph signals with change-points.
        
        #### Input.
        # size: lenght of the signal to be generated. 
        
        
        x=self.generators[0].generate_signal(self.change_points[0])
        self.history.append(x)
               
        for i in range(1,len(self.generators)):           
            x=self.generators[i].generate_signal(self.change_points[i]-self.change_points[i-1],**kargs)
            self.history.append(x)
            
         
        return np.vstack(self.history)
                            
    
        

class stationary_graph_signal_simulator():
    ##### Class generating different GS with different graph structure 
    def __init__(self,n_nodes,ux,spectral_profile,type_graph="erdos",type_noise="gaussian",save_history=True,seed=None,**kargs):
        
        
        ####### The algorithm asks the user a spectral_profile for the Graph filter that will be applied 
        ### to white noise so the output signal is stationary with respect to the graph structure. 
        
          
        
        #### This section initialize the structure of the graph signal
        
        ### Input
        # n_nodes=number of nodes
        # ux= mean of the signal
        # spectral_profile= a function generating the power spectral density of the sygnal. 
        # type_graph=available  graph model (erdos,euclidean,barabasi_albert)
        # type_noise = distribution of the noise (gaussian,uniform) that will be used to ge
        
        self.n_nodes=n_nodes
        self.type_graph=type_graph
        self.type_noise=type_noise
        self.ux=ux
        self.spectral_profile=spectral_profile
        self.seed=seed
        
     
        
        if type_graph=="erdos":
            
            self.G=graphs.ErdosRenyi(self.n_nodes, kargs['p'],seed=self.seed)
            self.G.set_coordinates()

            
       
        if type_graph=="barabasi_albert":
            self.G=graphs.BarabasiAlbert(self.n_nodes,m=kargs['m'],m0=kargs['m'],seed=self.seed)
            self.G.set_coordinates()
            
        if type_graph=="Minnesota":
            self.G=graphs.Minnesota()
       

       
        self.generate_fourier()
        self.generate_filter()
            
            
        
    def generate_fourier(self):
        #### Function to compute the Graph Fourier Transfort. 
        
        
        self.G.compute_fourier_basis()
        
       
    def generate_filter(self):
        #### This function defines the Filter that will be applied to withe noise. 
        
        self.H=filters.Filter(self.G, lambda x: self.spectral_profile(x))
        self.PSD=(self.spectral_profile(self.G.e))**2
        
        
    def generate_signal(self,size=1):
        
        #### Function to generate a stram of GS of a given size
        
        ### Input
        ## size= Number of GS to generate
        
        ### Output
        ## signal= generated signal 
        
        
       np.random.seed(self.seed)
       
       if self.type_noise=="gaussian":         
           e=np.random.normal(size=(self.n_nodes,size))
        
       if self.type_noise=="uniform":
            e=np.random.uniform(-np.sqrt(3),np.sqrt(3),size=(self.n_nodes,size))
    
       if self.type_noise=="t":         
            e=np.random.standard_t(df=100, size=(self.n_nodes,size))
        
       self.signal=self.H.filter(e).transpose()+self.ux
     
       return(self.signal)
                            

        
    def plot_signal(self,t=None):
        #### Function to prlot GS at a given time-stamp t, if t is None , the last observation is given 
        
        ## t:time point to plot
        if t is None:
            t=len(self.signal)-1
        self.G.plot_signal(self.signal[t], vertex_size=30)





