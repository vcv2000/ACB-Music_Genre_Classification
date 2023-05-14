# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:01:54 2022

@author: luisp
"""
import pandas as pd
import numpy as np
import os


def calculate_metrics(true_data, pred_data, metric=['explained_variance_score',
                                         'max_error',
                                         'mean_absolute_error',
                                         'mean_squared_error',
                                         'mean_squared_log_error',
                                         'mean_absolute_percentage_error',
                                         'median_absolute_error',
                                         'r2_score',
                                         'pearson_correlation_coefficient',
                                         'spearman_correlation_coefficient'],
                      power=0, write_mode=False, output_name=''):
    from sklearn.metrics import explained_variance_score, max_error,mean_absolute_error, mean_squared_error, mean_squared_log_error,mean_absolute_percentage_error, median_absolute_error, r2_score
    from scipy import stats
    
    dic={'explained_variance_score': explained_variance_score,
         'max_error':max_error,
         'mean_absolute_error': mean_absolute_error,
         'mean_squared_error': mean_squared_error,
         'mean_squared_log_error':mean_squared_log_error,
         'mean_absolute_percentage_error':mean_absolute_percentage_error,
         'median_absolute_error':median_absolute_error,
         'r2_score':r2_score,
         'pearson_correlation_coefficient':stats.pearsonr ,
         'spearman_correlation_coefficient':stats.spearmanr}
    
    dic_2={}
    for i in metric:
        if i!='pearson_correlation_coefficient' and 'spearman_correlation_coefficient':
            if i=='mean_squared_log_error':
                try:
                    dic_2[i]=dic[i](true_data, pred_data)
                except:
                    dic_2[i]=None
            else:
                dic_2[i]=dic[i](true_data, pred_data)
        if i=='pearson_correlation_coefficient':
            r,p=stats.pearsonr(np.squeeze(true_data),np.squeeze(pred_data))
            dic_2['pearson_correlation_coefficient']=r
        if i=='spearman_correlation_coefficient':
            rho,pval=stats.spearmanr(true_data, pred_data)
            dic_2['spearman_correlation_coefficient']=rho
    if write_mode==True:   
        output_file=open(output_name,'w')
        output_file.write('Metric; Value \n')
        for result in list(dic_2.keys()):
            if dic_2[result]==None:
                output_file.write(result + ';' + str(dic_2[result]) + '\n')
            else:
                output_file.write(result + ';' + str(np.round(dic_2[result],3)) + '\n')
    
    return dic_2
            



