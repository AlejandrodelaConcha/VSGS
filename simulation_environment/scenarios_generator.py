# -----------------------------------------------------------------------------------------------------------------
# Title:  scenatios_generator
# Author(s):      
# Initial version:  [insert date in 20200320]
# Last modified:    [insert date in 20200319]
# Designed for:     Python 3.7.0, macOS Catalina version 10.15.1
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This script generates the SGS according to the scenarios described in the paper.
# -----------------------------------------------------------------------------------------------------------------
# Notes: This algorithm requires graph.signals.generator
# -----------------------------------------------------------------------------------------------------------------
# Required libraries :  numpy, scipy,copy
# -----------------------------------------------------------------------------------------------------------------
# Key words:Graph signal processing,change-point dectection.
# -----------------------------------------------------------------------------------------------------------------

import numpy as np 
from simulation_environment.graph_signals_generator import *
import copy
import scipy
from scipy.stats import gamma



def Scenario_I(n_nodes,mean_change_points,fixed_frequencies,random_frequencies,mean_exponential,p_erdos,seed=None,**kargs):
       
    ##### Function generating a stream of graph signals according to Scenario I.    
   
    ##### Input. 
    # 1) The number of change-points is generated via a Poisson distribution with mean "mean_change-points".
    # 2) The distance between change-points is at least 30 + an exponential distribution with mean "mean_exponential".
    # 3) An Erdos Graph with "n_nodes" and probability "p_erdos" is generated. 
    # 4) The spectral profile of the filter is np.sqrt(15)/(np.log(x+10)+1).
    # 5) The lowest "fixed_frequencies" are generated at random previous the first change-point. 
    # 6) A given number "random_frequencies" are selected at random and they are modified after each change-point.
    

    ##### Output. 
    ## signal= dataset of the stream of graph_signals.
    ## GSO= Graph Shift Operator.
    ## change_points= list with the change-points location.
    ## mu= means of the signals in the espectral domain. 
    ## PSD= Power Specctral Density of the signal. 
    ## G_erdos_H0.G= Graph Structure. 
    
    np.random.seed(seed)
    
    generators=[]
    mu=[]
    change_points=[]
    
    ### Genarates the number of change-points.
    
    n_change_points=np.random.poisson(lam=mean_change_points)
    
    change_point=int(np.random.exponential(scale=mean_exponential)+30)
        
    #### Generates the change-points  location   
    change_points.append(change_point)
    for i in range(1,n_change_points):
        change_points.append(int(change_points[i-1]+np.random.exponential(scale=mean_exponential)+30))
           
    T=int(change_points[len(change_points)-1]+np.random.exponential(scale=mean_exponential)+30) 
    change_points.append(T)
    
    #### Generationg the graph structure 
    spectral_profile=lambda x: np.sqrt(15)/(np.log(x+10)+1)
    G_erdos_H0=stationary_graph_signal_simulator(n_nodes=n_nodes,ux=np.zeros(n_nodes),spectral_profile=spectral_profile,type_graph="erdos",p=p_erdos,seed=seed,**kargs) 

    
    ############### Generates the means of the Stream.
    
    ### Generates the first mean of the signal. 
    mu.append(np.concatenate((np.random.uniform(-5.0,5.0,size=fixed_frequencies),np.zeros(n_nodes-fixed_frequencies))))
    
    ##### Generates the mean of the signals after the change-points
    for i in range(1,n_change_points+1):
        frecuencies=np.random.choice(range(n_nodes),size=random_frequencies,replace=False)
        aux_mu=copy.copy(mu[0])
        aux_mu[frecuencies]=np.random.uniform(-5.0,5.0,size=random_frequencies)
        mu.append(aux_mu)
    
   
    GSO=G_erdos_H0.G.L.todense()
    lamb,U=scipy.linalg.eigh(GSO)
    PSD=G_erdos_H0.PSD
  
    for i in range(0,n_change_points+1):  
        G_erdos_H=copy.deepcopy(G_erdos_H0)
        G_erdos_H.ux=U.dot(mu[i])
        generators.append(G_erdos_H)

    ##### Generates the Stream of Graph Signals with change-points. 
  
    change_point_erdos=change_point_generator(generators,change_points)
    signal=change_point_erdos.generate_signal()
    
    
    return signal,GSO,change_points,mu,PSD,G_erdos_H0.G
    

def Scenario_II(n_nodes,fixed_frequencies,biggest_nodes,random_nodes,mean_exponential,m_barabasi=4,seed=None,**kargs):
    
   ##### Function generating a stream of graph signals according to Scenario II: 
   
    ##### Input. 
    # 1) 4 change_points are generated (This includes the final point of the segment)
    # 2) The distance between change-points is at least 30 + an exponential distribution with mean "mean_exponential".
    # 3) An Barabasi Albert graph with "n_nodes" and each incomming node is connected with "m_barabasi" nodes.
    # 4) The spectral profile of the filter is 2*gamma.pdf(x,a=20.,loc=5.)+1.0.
    # 5) The lowest "fixed_frequencies" are generated at random previous the first change-point.  
    # 6) After the first change-point the node with the highest degree and all its neigbors modify their mean. 
    # 7) After the second change-point the k nodes with the highest degree modify their mean, where k="biggest_nodes"
    # 8) After the third change-points k nodes modify their mean, where k="random_nodes"
    
    ##### Output. 
    ## signal= dataset of the stream of graph_signals.
    ## GSO= Graph Shift Operator.
    ## change_points= list with the change-points location.
    ## mu= means of the signals in the espectral domain. 
    ## PSD= Power Specctral Density of the signal. 
    ## G_barabasi_albert_H0.G= Graph Structure. 
    
    np.random.seed(seed)
    
    
    generators=[]
    change_points=[]
    
    #### generating the change-points    
    change_point=int(np.random.exponential(scale=mean_exponential)+30)
    change_points.append(change_point)
    for i in range(1,3):
        change_points.append(int(change_points[i-1]+np.random.exponential(scale=mean_exponential)+30))
    
    T=int(change_points[len(change_points)-1]+np.random.exponential(scale=mean_exponential)+30) 
    change_points.append(T)
    #### Generating the graph structure 
    spectral_profile=lambda x: 2*gamma.pdf(x,a=20.,loc=5.)+1.0
    
    G_barabasi_albert_H0=stationary_graph_signal_simulator(n_nodes,ux=np.zeros(n_nodes),spectral_profile=spectral_profile,type_graph="barabasi_albert",m=m_barabasi,seed=seed,**kargs)    
 
    GSO=G_barabasi_albert_H0.G.L.todense()
    lamb,U=scipy.linalg.eigh(GSO)
    PSD=G_barabasi_albert_H0.PSD
    
    
    #### Generating the means.
    
    ### Before the first change-point. 
    mu=np.concatenate((np.random.uniform(-5.,5.,size=fixed_frequencies),np.zeros(n_nodes-fixed_frequencies)))
    G_barabasi_albert_H0.ux=U.dot(mu)
    

    G_barabasi_albert_H1=copy.deepcopy(G_barabasi_albert_H0)
   
 
    
    ## After the first change-point.
    biggest=np.argmax(np.diag(GSO))
    index=np.where(GSO[biggest]!=0)[1]
    G_barabasi_albert_H1.ux[index]=np.random.uniform(-5.,5.,size=len(index))
    
    
    ## After the second change-point. 
    G_barabasi_albert_H2=copy.deepcopy(G_barabasi_albert_H1)
    index=np.where(np.argsort(np.diag(GSO))>n_nodes-biggest_nodes-1)[0]
    print(len(index))
    G_barabasi_albert_H2.ux[index]=np.random.uniform(-5.,5.,size=len(index))

    
    ## After the third change-point. 
    G_barabasi_albert_H3=copy.deepcopy(G_barabasi_albert_H2)
    np.random.seed(seed)
    index=np.random.choice(n_nodes,size=random_nodes,replace=False)
    index=[x for x in index]
    G_barabasi_albert_H3.ux[index]=np.random.uniform(-5.,5.,size=len(index))
         
    generators=[G_barabasi_albert_H0,G_barabasi_albert_H1,G_barabasi_albert_H2,G_barabasi_albert_H3]
        
    mu=[]
   
    for i in range(len(generators)):
        mu.append(generators[i].ux)

    #### generating the signal 
    change_point_barabasi=change_point_generator(generators,change_points)
    signal=change_point_barabasi.generate_signal()
    
    return signal,GSO,change_points,mu,PSD,G_barabasi_albert_H0.G
    
def Scenario_III(fixed_frequencies,number_hops,random_nodes,min_distance,mean_exponential,seed=None,**kargs):
    
   ##### Function generating a stream of graph signals according to Scenario II: 
   
    ##### Input. 
    # 1) 4 change_points are generated (This includes the final point of the segment)
    # 2) The distance between change-points is at least 30 + an exponential distribution with mean "mean_exponential".
    # 3) An Barabasi Albert graph with "n_nodes" and each incomming node is connected with "m_barabasi" nodes.
    # 4) The spectral profile of the filter is 2*gamma.pdf(x,a=20.,loc=5.)+1.0.
    # 5) The lowest "fixed_frequencies" are generated at random previous the first change-point.  
    # 6) After the first change-point the node with the highest degree and all its neigbors modify their mean. 
    # 7) After the second change-point the k nodes with the highest degree modify their mean, where k="biggest_nodes"
    # 8) After the third change-points k nodes modify their mean, where k="random_nodes"
    
    ##### Output. 
    ## signal= dataset of the stream of graph_signals.
    ## GSO= Graph Shift Operator.
    ## change_points= list with the change-points location.
    ## mu= means of the signals in the espectral domain. 
    ## PSD= Power Specctral Density of the signal. 
    ## G_barabasi_albert_H0.G= Graph Structure. 
    
    np.random.seed(seed)
    
    
    generators=[]
    change_points=[]
    
    #### generating the change-points    
    change_point=int(np.random.exponential(scale=mean_exponential)+min_distance)
    change_points.append(change_point)
    for i in range(1,2):
        change_points.append(int(change_points[i-1]+np.random.exponential(scale=mean_exponential)+min_distance))
    
    T=int(change_points[len(change_points)-1]+np.random.exponential(scale=mean_exponential)+min_distance) 
    change_points.append(T)
    #### Generating the graph structure 
    spectral_profile=lambda x: np.sqrt(15)/(np.log(x+1)+1)
    
    n_nodes=2642
    
    G_minnesota_H0=stationary_graph_signal_simulator(n_nodes=n_nodes,ux=np.zeros(n_nodes),spectral_profile=spectral_profile,type_graph="Minnesota",seed=seed,**kargs)    
    
    GSO=G_minnesota_H0.G.L.todense()
    lamb,U=scipy.linalg.eigh(GSO)
    PSD=G_minnesota_H0.PSD
    
   
    W_2=G_minnesota_H0.G.W.dot(G_minnesota_H0.G.W)
    W_3=G_minnesota_H0.G.W.dot(W_2)
    W_4=G_minnesota_H0.G.W.dot(W_3)
    W_5=G_minnesota_H0.G.W.dot(W_4)
    
    aux_W=G_minnesota_H0.G.W+W_2+W_3+W_4+W_5
    aux_W=aux_W.todense()

    #### Generating the means.
    
    ### Before the first change-point. 
    mu=np.concatenate((np.random.uniform(-5.,5.,size=fixed_frequencies),np.zeros(n_nodes-fixed_frequencies)))
    G_minnesota_H0.ux=U.dot(mu)
   
    G_minnesota_H1=copy.deepcopy(G_minnesota_H0)

    ## After the first change-point.
    index=np.random.randint(n_nodes,size=number_hops)
    
    bernoulli=np.random.binomial(size=number_hops,n=1, p= 0.5)
    
    G_minnesota_H1.ux[index]+=(bernoulli-1.*(1.-bernoulli))*np.random.uniform(1.,5.,size=len(index))

    for i in range(len(index)):
        three_hop=np.where(aux_W[index[i],:]>0)[1]
        G_minnesota_H1.ux[three_hop]+=(bernoulli[i]-1.*(1.-bernoulli[i]))*np.random.uniform(1.,5.,size=len(three_hop))
        
        
    ## After the third change-point. 
    G_minnesota_H2=copy.deepcopy(G_minnesota_H1)
    np.random.seed(seed)
    index=np.random.choice(n_nodes,size=random_nodes,replace=False)
    index=[x for x in index]
    
    bernoulli=np.random.binomial(size=random_nodes,n=1, p= 0.5)
    G_minnesota_H2.ux[index]+=(bernoulli-1.*(1.-bernoulli))*np.random.uniform(5.,10.,size=len(index))
         
    generators=[G_minnesota_H0,G_minnesota_H1,G_minnesota_H2]
        
    mu=[]
   
    for i in range(len(generators)):
        mu.append(generators[i].ux)

    #### generating the signal 
    change_point_minnesota=change_point_generator(generators,change_points)
    signal=change_point_minnesota.generate_signal()
    
    return signal,GSO,change_points,mu,PSD,G_minnesota_H0.G










