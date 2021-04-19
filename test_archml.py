import pandas as pd
import numpy as np
import os

from archml.helpers import resizer, result_metrics, compare

def test_resizer():
	resizer(data="ModelsAndSampleData/100samples.hdf5",csv="ModelsAndSampleData/100samples.csv",size=50)
	csv100 = pd.read_csv("ModelsAndSampleData/100samples.csv")
	csv50 = pd.read_csv("steadmini_50.csv")

	assert len(csv50) == len(csv100)/2

	os.remove("steadmini_50.csv")
	os.remove('steadmini_50.hdf5')


def test_results_metrics():

	eqt = pd.read_csv("ModelsAndSampleData/X_test_results_EQT.csv")
	test_dict = {'EQT':eqt}
	test_columns = np.array(["model_name","det_recall","det_precision","d_tp","d_fp","d_tn","d_fn","p_recall","p_precision","p_mae","p_rmse","p_tp","p_fp","p_tn","p_fn","s_recall","s_precision","s_mae","s_rmse","s_tp","s_fp","s_tn","s_fn","#events","#noise"])
	result_metrics(test_dict)
	
	csv = pd.read_csv("test_results.csv")
	
	comparison = csv.columns == test_columns
	equal_arrays = comparison.all()

	assert equal_arrays

	os.remove('test_results.csv')

	
def test_compare():
	csv = pd.read_csv("ModelsAndSampleData/test_results.csv")
	a = compare(csv)
	assert a == 'det_recall'



