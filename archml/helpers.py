import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def resizer(data,csv,size):

	#STEAD downsizer like code
	
    #read csv and hdf5 files
    csv = pd.read_csv(csv)
    stead = h5py.File(data,'r')
    
    #take the group object inside the file
    chunk = stead['data']
    
    #shuffle csv to randomize trace_name order and save the resultig csv
    shuffled_csv = csv.sample(frac=1).reset_index(drop=True)
    shuffled_csv[:len(shuffled_csv)//(100//size)].to_csv("steadmini_{}.csv".format(size))
    
    #export the 10% fo trace names out of the shuffled csv 
    ev_list = shuffled_csv['trace_name'].to_list()[:int(len(chunk)//(100//size))] 
    
    print(int(len(chunk)//(100//size)))
    print(len(ev_list))
    #create a new hdf5 file
    steadmini_c = h5py.File('steadmini_{}.hdf5'.format(size),'w')
    
    #create a group named 'data' inside the file
    small_chunk = steadmini_c.create_group("data")
    
    #copy the files from the bigger chunk to the new file using the downsized name list(ev_list)
    for c, evi in enumerate(ev_list):
        chunk.copy(evi,small_chunk)
        
    #cclose the file to save the new file
    steadmini_c.close()


def visualizer(model, input_hdf5, input_csv, layers_to_visualize):

	#use EQT Paper task code

	return graphs

def result_metrics(models_and_test_results_csv_dict):

	#use Test2Results code
	models = models_and_test_results_csv_dict
	models_test = pd.DataFrame()


	for i in models.keys():
	    csv = []
	    #print("Model : {}".format(i))  
	    metrik = ["detection_probability","P_probability","S_probability"]
	    csv.append(i)
	    for j in metrik:
	        TP=0
	        FP=0
	        TN=0
	        FN=0
	        event = 0
	        noise = 0
	        for a in range(len(models[i])):
	            if models[i]["trace_category"][a] == "earthquake_local":
	                event +=1
	                if np.isnan(models[i]["{}".format(j)][a]):
	                    FN +=1
	                else:
	                    TP +=1
	            elif models[i]["trace_category"][a] == "noise":
	                noise +=1
	                if np.isnan(models[i]["{}".format(j)][a]):
	                    TN +=1
	                else:
	                    FP +=1

	                 
	        #print(j,"\n","TP",TP,"\n","TN",TN,"\n","FP",FP,"\n","FN",FN,"\n","Total events",event,"\n" "Total noise",noise)
	        
	        recall = TP/(TP+FN)
	        precision = TP/(TP+FP)
	        
	        df = models[i].dropna()
	        #print("Recal", recall,"\n", "Precision",precision)
	        csv.append(recall)
	        csv.append(precision)
	        
	        if j == "P_probability":
	            mae = mean_absolute_error(df["P_pick"],df["p_arrival_sample"])**(1/2)
	            #print("MAE P:{}".format(mae))
	            rmse = mean_squared_error(df["P_pick"],df["p_arrival_sample"])**(1/2)
	            #print("RMSE P:{}".format(rmse))
	            csv.append(mae)
	            csv.append(rmse)
	            
	        if j == "S_probability":   
	            mae = mean_absolute_error(df["S_pick"],df["s_arrival_sample"])**(1/2)
	            #print("MAE S:{}".format(mae))
	            rmse  =mean_squared_error(df["S_pick"],df["s_arrival_sample"])**(1/2)
	            #print("RMSE S:{}".format(rmse))
	            csv.append(mae)
	            csv.append(rmse)
	    
	        csv.append(TP)
	        csv.append(FP)
	        csv.append(TN)
	        csv.append(FN)
	        
	            
	        
	        #print("\n")
	    
	    #print(csv)
	    #print("\n")
	    csv.append(event)
	    csv.append(noise)
	    models_test = models_test.append(pd.DataFrame(csv).T)
	test_columns = ["model_name","det_recall","det_precision","d_tp","d_fp","d_tn","d_fn","p_recall","p_precision","p_mae","p_rmse","p_tp","p_fp","p_tn","p_fn","s_recall","s_precision","s_mae","s_rmse","s_tp","s_fp","s_tn","s_fn","#events","#noise"]
	models_test.columns = test_columns
	model_test = models_test.reset_index()
	model_test = model_test.drop("index",axis=1)
	model_test.to_csv("test_results.csv",index=False) 


def compare(result_catalog_csv, model_to_compare_to='EQT'):

	#use EQT Paper code

	b = result_catalog_csv

	name_count = 0
	good_columns = ["det_recall","det_precision","d_tp","d_tn","p_recall","p_precision","p_tp","p_tn","s_recall","s_precision","s_tp","s_tn"]
	bad_columns = ["d_fp","d_fn","p_fp","p_fn","s_fp","s_fn","p_mae","p_rmse","s_mae","s_rmse"]
	for name in b.model_name:
	    #print(name)
	    better_list = []
	    for column in b.columns:
	        if column in good_columns:
	            if b[column][name_count] > b[column][0]:
	                better_list.append(column)
	        elif column in bad_columns:    
	            if b[column][name_count] < b[column][0]:
	                better_list.append(column)
	    #print(better_list)
	    name_count += 1
	    print("{} performed better than EQT tranied w/ STEADmini on {} parameters:".format(name,len(better_list)))
	    for parameter in better_list:
	          print(parameter)
	return parameter




