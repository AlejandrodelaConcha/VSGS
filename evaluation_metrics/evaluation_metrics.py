# -----------------------------------------------------------------------------------------------------------------
# Title: valuation_metrics
# Author(s):      Alejandro de la Concha   
# Initial version:  [insert date in 20200508]
# Last modified:    [insert date in 20200508]
# Designed for:     Python 3.7.0, macOS Catalina version 10.15.1
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): This algorithm returns the valuation metrics for the results of a change-poiint detection algorithm.
# -----------------------------------------------------------------------------------------------------------------
# Notes:
# -----------------------------------------------------------------------------------------------------------------
# Required libraries : ruptures,numpy
# -----------------------------------------------------------------------------------------------------------------
# Key words:change-point detection
# -----------------------------------------------------------------------------------------------------------------


from ruptures.metrics import *
import numpy as np


def get_valuation_metrics(real_change_points,estimated_change_points):
    ### This function computes the valuation metrics for the results of a change-poiint detection algorithm.
    
    #### Input.
    ## real_change_points: real change points. 
    ## estimated_change_points: change-points given by a change-point detection algorithm. 
    
    #### Output.
    ## haussdor distance
    ## randindex
    ## precision
    ## recall 
    ## F1
    
    if len(estimated_change_points)>1:
        haussdorf_distance=hausdorff(real_change_points,estimated_change_points)
    else:
        haussdorf_distance=np.nan
        
    randindex_=randindex(real_change_points,estimated_change_points)    
    precision,recall=precision_recall(real_change_points,estimated_change_points)

    if (precision+recall)==0:
        F1=0.0
    else:
        F1=2.*(precision*recall)/(precision+recall)
    
    return haussdorf_distance,randindex_,precision,recall,F1