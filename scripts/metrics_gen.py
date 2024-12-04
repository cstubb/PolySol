import pickle
from copy import deepcopy
from itertools import chain,repeat
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from pathlib import Path

#  dff - Short for dataframe(df) final

class modelGroup:
	""" 
	Abstract class for parsing groups of trained ML model output. ,
 	Instantiate with a group_id (arbitrary), a group_dir directory containing the model runs, a model_num_str model number string (usually 2345 - identifies which architectures were run),
    a list of short names without spaces, a list of longer names for plotting, and whether PCA was used (always False)
 	
    Includes methods for 
	- loading model data (load_model_data method),
 	- setting y_data (set_y_data method), 
    - generating model metrics (gen_sklearn_metrics method), 
    - and comparing different model runs (gen_dff method)
 
 
	Example usage:
		mod_nums = "2345"
		codi_short_names = ['co_di_atom_bd', 'co_di_mordred', 
							'co_di_mfp', 'co_di_rdfp', 'co_di_atom_mordred',
							'co_di_atom_morganfp', 'co_di_atom_rdfp'] # BY DESIRED ORDER
		codi_long_names = ['Atom and Bond (CoDi)', 
										'Mordred (CoDi)', 
										'Morgan FP (CoDi)', 
										'RDKit FP (CoDi)',
										'Atom + Mordred (CoDi) ', 
										'Atom + Morgan FP (CoDi)', 
										'Atom + RDKit FP (CoDi)'] 
		codi_mout = modelGroup(group_id = 'co_di', 
  								 group_dir = "/a/dir", 
  								 model_num_str = mod_nums, 
           						 model_names_short = codi_short_names, 
                  				 descriptor_names_long = codi_long_names,
								 pca_used = False,
         						)
		codi_mout.load_model_data(hush=False)
		codi_mout.set_y_data(f"{codi_dir + mod_nums}_copoly_y_train_data.pkl",
								f"{codi_dir + mod_nums}_copoly_y_test_data.pkl")
		codi_mout.gen_sklearn_metrics(metrics_list=metrics_name_list)
		codi_mout.gen_dff(hush=True)
		dff_codi = codi_mout.dff # Use to examine model performance
  		display(dff_codi)
	"""
	def __init__(self, group_id: str, group_dir: str, model_num_str: str, 
				model_names_short: list, descriptor_names_long: str, pca_used: bool,
    			rand_seed_used):
		self.group_id = group_id
		self.group_dir = Path(group_dir)
		self.model_num_str = model_num_str
		self.model_names_short = model_names_short
		self.descriptor_names_long = descriptor_names_long
		self.pca_used = pca_used
		self.rand_seed = rand_seed_used
	


	def load_model_data(self, hush: bool=False, pth_overrides = None):
		#if not hush:
		print(f"Loading data for {self.group_id}...")
		pca_string = "NOPCA" if self.pca_used is False else ""
		
		model_list_paths = []
		kfold_list_paths = []
		X_train_paths = []
		X_test_paths = []

		for mname in self.model_names_short:
			if self.pca_used == True:
				model_list_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_models_list.pkl')
				kfold_list_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_kfold_list.pkl')
				X_train_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_X_train.pkl')
				X_test_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_X_test.pkl')
			else:
				model_list_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_{pca_string}_rs{self.rand_seed}_models_list.pkl')
				kfold_list_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_{pca_string}_rs{self.rand_seed}_kfold_list.pkl')
				X_train_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_{pca_string}_rs{self.rand_seed}_X_train.pkl')
				X_test_paths.append(self.group_dir/f'{mname}_{self.model_num_str}_{pca_string}_rs{self.rand_seed}_X_test.pkl')
		
		# If an override list with entries formatted as
		# 		[mlist_pth, kfold_pth, xtrain_pth, xtest_pth] <--- per set of descriptor
		# is provided, 
		if pth_overrides is not None:
			# Yes, this is stupid. 
			# Yes, it works
			model_list_paths = []
			kfold_list_paths = []
			X_train_paths = []
			X_test_paths = []
			for override_list in pth_overrides:
				mlist_pth, kfold_pth, xtrain_pth, xtest_pth = override_list
				model_list_paths.append(mlist_pth)
				kfold_list_paths.append(kfold_pth)
				X_train_paths.append(xtrain_pth)
				X_test_paths.append(xtest_pth)

		all_model_lists = []
		all_kfold_lists = []
		all_X_train_data = []
		all_X_test_data = []
		i = 0
		for mod_path, kfold_path, X_train_path, X_test_path in zip(model_list_paths,
																	 kfold_list_paths,
																	 X_train_paths,
																	 X_test_paths):
			if not hush:
				print(f"\tLoading the {i}th model...")
			mod_name = self.model_names_short[i] #! Excluded PCA from names

			with open(mod_path, 'rb') as f:
				pkl_out = pickle.load(f)
				all_model_lists.append((mod_name, pkl_out))
				if not hush:
					print(f"\t\t{mod_name} model data loaded sucessfully")

			with open(kfold_path, 'rb') as f:
				pkl_out = pickle.load(f)
				all_kfold_lists.append((mod_name, pkl_out))
				if not hush:
					print(f"\t\t{mod_name} kfold data loaded sucessfully")

			with open(X_train_path, 'rb') as f:
				pkl_out = pickle.load(f)
				all_X_train_data.append((mod_name, pkl_out))
				if not hush:
					print(f"\t\t{mod_name} X train loaded sucessfully")

			with open(X_test_path, 'rb') as f:
				pkl_out = pickle.load(f)
				all_X_test_data.append((mod_name, pkl_out))
				if not hush:
					print(f"\t\t{mod_name} X test data loaded sucessfully")
			i += 1

		self.all_model_lists = all_model_lists # Changed
		self.all_kfold_lists = all_kfold_lists # Changed
		self.all_X_train = all_X_train_data
		self.all_X_test = all_X_test_data
		if not hush:
			print("\tSuccesfully loaded all model data.\n")

		#* Below is completely redundant and can be safely removed at any time. 
		if not hush:
			print(f"\tGenerating dataframe for {self.group_id}...")
		df = pd.DataFrame(self.all_kfold_lists).T 
		# Explode all columns
		df = df.explode(column=list(df.columns))
		# Converts dataframe into long form by melting 
		df = df.T.melt(id_vars=0, value_vars=list(df.T.columns)[1:])
		if 'model_name' not in df.columns: # Likely not necessary
			# Create columns from melt list
			df = df.assign(model_name=[x[0] for x in df.value],#Technically architecture name
							model_function=[x[1] for x in df.value],
							test_acc_average=[x[3] for x in df.value],
							kfold_acc_average=[x[2] for x in df.value],
							kfold_array=[x[4] for x in df.value])
			# Remove unnecessary melt columns
			df = df.drop(columns=['variable','value'])
			# Rename columns
			df.columns = ['descriptor_name'] + list(df.columns)[1:]
			df = df.sort_values('descriptor_name')
			df = df.reset_index(drop=True)
		df = df.sort_values('descriptor_name')
		self.models_df = df
		if not hush:
			print(f"\t\tSuccesfully generated dataframe from model output for {self.group_id}")
		
		print("    ...done")



	def set_y_data(self, y_path_train, y_path_test):
		with open(y_path_train, 'rb') as f:
			self.y_train = pickle.load(f)
			#print("\tSuccesfully loaded y train from: ",y_path_train)
		with open(y_path_test, 'rb') as f:
			self.y_test = pickle.load(f)
			#print("\tSuccesfully loaded y test from: ",y_path_test)



	def gen_sklearn_metrics(self, metrics_list):
		print("\tGenerating scikit-learn metrics (and confusion matrices)...")
		if type(metrics_list) != list or metrics_list is None:
			print("Warning: metrics_list is not a list.")
			pass
		if not isinstance(self.all_model_lists, list):
			print("Warning: All model lists is not a list.")
			pass
		if len(self.all_model_lists) == 0:
			print("Warning: length of all_model_lists is zero.")
			pass
			

		all_cms = []
		all_metrics_dicts = {}
		for i, lst in enumerate(self.all_model_lists):
			descrip_name, sublst = lst
			X_test_tmp = self.all_X_test[i][1]

			tmp_cm_lst = []
			tmp_metrics_superdict = {}
			for entry in sublst:
				(archi_name, classif, archi_acc) = entry
				y_pred_classif = classif.predict(X_test_tmp)
				cmatrix = confusion_matrix(self.y_test, y_pred_classif, labels=classif.classes_)
				cmatrix_display = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=classif.classes_)
				tmp_cm_lst.append([archi_name, cmatrix, cmatrix_display])

				tmp_metrics_dict = {}
				for metric in metrics_list:
					metric_name, metric_func = metric
					tmp_metrics_dict[metric_name] = metric_func(self.y_test, y_pred_classif)				

				tmp_metrics_superdict[archi_name] = tmp_metrics_dict
			all_cms.append([descrip_name, tmp_cm_lst])

			all_metrics_dicts[descrip_name] = tmp_metrics_superdict
			
			
		self.all_cms = all_cms
		all_metrics_dicts = dict(all_metrics_dicts)
		self.all_metrics_dicts = all_metrics_dicts
		print("\t...done")



	def gen_dff(self, hush: bool=True):
		""" 
		After generating initial metrics and setting a metrics dict, we generate a final dataframe.
		This is termed 'dff' (self.dff)
		"""

		if not hush:
			print(f"\tGenerating dataframe for {self.group_id}...")




		df_initial = pd.DataFrame(self.all_kfold_lists).T 
		# Explode all columns
		df_initial = df_initial.explode(column=list(df_initial.columns))
  
		# Converts dataframe into long form by melting 
		df_initial = df_initial.T.melt(id_vars=0, value_vars=list(df_initial.T.columns)[1:])
		if 'model_name' not in df_initial.columns: # Likely not necessary
			# Create columns from melt list
   
			if not hush:
				print("KFold acc 'averages':",[x[2] for x in df_initial.value])
			df_initial = df_initial.assign(model_name=[x[0] for x in df_initial.value],
							model_function=[x[1] for x in df_initial.value],
							test_acc_average=[x[3] for x in df_initial.value],
							kfold_acc_average=[x[2] for x in df_initial.value],
							kfold_array=[x[4] for x in df_initial.value])
			# Remove unnecessary melt columns
			df_initial = df_initial.drop(columns=['variable','value'])
			# Rename columns
			df_initial.columns = ['descriptor_name'] + list(df_initial.columns)[1:]
			df_initial = df_initial.sort_values('descriptor_name')
			df_initial = df_initial.reset_index(drop=True)
		df_initial = df_initial.sort_values('descriptor_name')
		df_initial = df_initial
		if not hush:
			print(f"\tSuccesfully generated dataframe from model output for {self.group_id}")
		

		#* Converts dictionaries to their own dataframes
		df = pd.DataFrame(self.all_metrics_dicts)


		#* Convert Series of dicts to dataframe, over all columns.
		df_lst = [df[f'{colname}'].apply(pd.Series) for colname in df.columns]
		df_mega = pd.concat(df_lst)
		df_mega = df_mega.reset_index().rename(columns={'index':'archi_tmp'})

		num_archi = len(self.model_num_str)
		repeat_names = [repeat(mod_name, num_archi) for mod_name in self.all_metrics_dicts]
		descrip_names = [x for x in chain.from_iterable(repeat_names)]
		df_mega.index = descrip_names


		#* Assign abbreviations to each architecture
		idents_dict = deepcopy(self.all_metrics_dicts)
		archi_name_dict = {
		    "Nearest Neighbors": "NN",
		    "Linear SVM": "SVM",
		    "Decision Tree": "DT",
		    "Random Forest": "RF",
		    "AdaBoost": "AB",
		    "Naive Bayes": "NB",
		}
		#* Assign labels to each model (based on descriptor short name + architecture abbreviation)
		for key,val_dict in idents_dict.items():
			for archi_key, _ in val_dict.items():
				# e.g. homo_nopca_atom_bd_RF
				val_dict[archi_key] = f"{self.group_id}_{key}_{archi_name_dict[archi_key]}"





		#* Generate labels for metrics dataframe (df_mega)
		#* Not too expensive since only ~20-40 rows.
		labels_list_mega = []
		for row in df_mega.itertuples():
			descrip = row.Index
			archi = row.archi_tmp
			label = idents_dict[descrip][archi]

			#* Check if group id is in label more than once. If so, changes label to only be encoded once.
			if label.count(self.group_id) > 1:
				print("Found model name to clean up - potential collision with model_name.")
				len_group_id = len(self.group_id) + 1
				print(f"{label} --> {label[len_group_id:]}")
				label = label[len_group_id:]

			labels_list_mega.append(label)
			if not hush:
				print("\t\t",descrip,archi,"--->",label)
    
		#* Generate labels for 'models' dataframe (models_df)
		labels_list_og = []
		for row in df_initial.itertuples():
			descrip = row.descriptor_name
			archi = row.model_name
			label = idents_dict[descrip][archi]
			#* Check if group id is in label more than once. If so, changes label to only be encoded once.
			if label.count(self.group_id) > 1:
				print("Found model name to clean up - potential collision with model_name.")
				len_group_id = len(self.group_id) + 1 # ** Plus one is important, accounts for '_' in name
				print(f"{label} --> {label[len_group_id:]}")
				label = label[len_group_id:]
			labels_list_og.append(label)
			if not hush:
				print("\t\t",descrip,archi,"--->",label)


		def print_when_nan(df, df_name):
			# Need two anys for DATAFRAMEs!. Logic is correct, but seems weird
			has_na_series = df.isna().any()
			nan_msg = f"\t{df_name} - No NaNs" if not has_na_series.any() else f"WARNING: {df_name} - NaN Values found!"
			nan_series_names = list(has_na_series[has_na_series.values == True].index)
			if has_na_series.any():
				print(f"WARNING: {self.group_id} has NaN values at {nan_series_names}")
			print(nan_msg)
		#* Add model label columns, then check for NaNs and a difference in accuracy per row.
		#* (diff in accuracy per row would indicate incorrect joining of dataframes)
		dfaa = df_initial.assign(model_label=labels_list_og).set_index('model_label')
		dfbb = df_mega.assign(model_label=labels_list_mega).set_index('model_label')
		print_when_nan(dfaa, 'dfaa')
		print_when_nan(dfbb, 'dfbb')
		dfcc = dfaa.join(dfbb)#, on='model')


		acc_diff = dfcc.test_acc_average - dfcc.accuracy
		if acc_diff.max() > 0.0001:
			print(f"\t\t\t~~~~WARNING~~~~~\n\t Accuracy mismatch found!({acc_diff.max():.3f})")

		print_when_nan(dfcc, 'dfcc')



		#* Add CMs to dataframe.
		#* This used to cause errors due to mismatching labels (and logic)
		labels_cms_list = []
		for (descrip,cm_lists) in self.all_cms:
			for (archi_name, cm_val, cm_disp) in cm_lists:

				label = f'{self.group_id}_{descrip}_{archi_name_dict[archi_name]}'
				#* Check if group id is in label more than once. If so, changes label to only be encoded once.
				if label.count(self.group_id) > 1:
					print("Found model name to clean up - potential collision with model_name.")
					len_group_id = len(self.group_id) + 1 # ** Plus one is important, accounts for '_' in name
					print(f"{label} --> {label[len_group_id:]}")
					label = label[len_group_id:]

				labels_cms_list.append([label,cm_val])
		df_cm = pd.DataFrame(labels_cms_list, columns=['model_label', 'cmatrix']).set_index('model_label') #.convert_dtypes()
		dff = dfcc.join(df_cm, on='model_label') #archi_tmp
		dff = dff.reset_index().drop(columns='archi_tmp')
		dff.insert(loc=3, column='id', value=self.group_id)
		nan_msg_f = "\tdff - No NaNs" if not dff.isna().any().any() else "WARNING: dff - NaN Values found!"
		print(nan_msg_f)
		if dff.isna().any().any():
			print("\t\t dff has NaN\n dfaa|dfbb|dfcc")
			print("Length:",(len(labels_list_og), len(labels_list_mega), len(labels_cms_list)))
			print(*zip(labels_list_og ,labels_list_mega, labels_cms_list), sep='\n')

		# Num Train, Num Test, Num Total, Num Features.
		# Be careful as num y datapoints may not always be same as # X datapoints
		dff = dff.assign(num_train=len(self.y_train))
		dff = dff.assign(num_test=len(self.y_test))
		dff = dff.assign(num_total=(len(self.y_train) + len(self.y_test)))
		dff = dff.assign(num_features_in= lambda df: [x.n_features_in_ for x in df.model_function.tolist()])
		

		dff = dff.assign(num_good_solv_train = self.y_train.sum())
		dff = dff.assign(num_bad_solv_train = (self.y_train.shape[0] - self.y_train.sum()))
		dff = dff.assign(num_good_solv_test = self.y_test.sum())
		dff = dff.assign(num_bad_solv_test = self.y_test.shape[0] - self.y_test.sum())
		dff = dff.assign(num_good_solv_total = lambda df: df.num_good_solv_train + df.num_good_solv_test)
		dff = dff.assign(num_bad_solv_total = lambda df: df.num_bad_solv_train + df.num_bad_solv_test)
			
		self.dff = dff
		




def _generate_individual_metrics_cm(models_list_func,metrics_to_calc,X_test_func,y_test_func):
	cm_list_func = []
	metrics_output = []
	for mod_list in models_list_func:
		name = mod_list[0]
		classif = mod_list[1]
		y_pred_classif = classif.predict(X_test_func)
		cmatrix = confusion_matrix(y_test_func, y_pred_classif, labels=classif.classes_)
		cmatrix_display = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=classif.classes_)
		cm_list_func.append([cmatrix,cmatrix_display])

		tmp_metrics_list = []
		for metr in metrics_to_calc:
			metric_name = metr[0]
			metric_function = metr[1]
			try:
				tmp_metrics_list.append([metric_name,metric_function(y_pred_classif,y_test_func)])
			except ValueError:
				print("Value error in metrics")
				tmp_metrics_list.append([metric_name,np.nan])

		metrics_output.append([name] + tmp_metrics_list)

	return metrics_output,cm_list_func


def generate_all_metrics_cm(all_model_lists, all_X_test_data, y_test_data, metrics_to_calc):
	""" 
	Takes a 3 model-specific lists (output, X test, y test) and a list of metrics (metrics_to_calc)
	Returns a list of output metrics and confusion matrices
	Should be refactored to use dataframe in future!
	"""
	all_metrics_out = []
	all_cm_out = []
	for i,mod_lst in enumerate(all_model_lists):
		mod_name = mod_lst[0]
		models_data = mod_lst[1]
		curr_X_test = all_X_test_data[i][1]
		curr_metrics_list, curr_cm_list = _generate_individual_metrics_cm(models_data, 
														metrics_to_calc, curr_X_test, 
														y_test_data) #! Make sure X,y data in correct format
		all_metrics_out.append([mod_name] + curr_metrics_list)
		all_cm_out.append([mod_name] + curr_cm_list)
	return all_metrics_out,all_cm_out

