import re
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import rdkit
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import rdMolDescriptors, Descriptors, rdmolfiles, PandasTools, AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D, MolDrawing, DrawingOptions
import mordred
from mordred import Calculator, descriptors
from itertools import combinations, permutations, pairwise,product,starmap
#from rdkit.Chem.Draw import IPythonConsole
import sklearn
import pickle
import json
from scipy import stats
from pathlib import Path
from natsort import natsorted



# Utility function
def _get_col_non_zero_percent(df, print_extra: bool=False, percent_cutoff: float=1.0):
	print("BEFORE NON ZERO DROPS",df.shape)
	non_zero_lst = []
	for col_name in df.columns:
		if (type(df[col_name][0])) == np.ndarray:
			print(col_name,"is array")
		elif (df[col_name] == 0).all():
			print("ALL ZERO",col_name)
		val_counts = df[col_name].value_counts()

		zero_counts = val_counts.index.isin([0.0])
		non_zero_counts = val_counts[~zero_counts]
		sum_val_counts = val_counts.sum()
		sum_nonzero_counts = non_zero_counts.sum()
		num_zero_counts = sum_val_counts - sum_nonzero_counts
		percent_non_zero = (sum_nonzero_counts/sum_val_counts)*100
		#! Added: Remove col if < cutoff
		if percent_non_zero < percent_cutoff:
			print(f"Dropped {col_name} as it was less than cutoff of {percent_cutoff}: {percent_non_zero}")
			df.drop(columns=col_name, axis=1, inplace=True)
			continue

		non_zero_lst.append((col_name,percent_non_zero))
		if print_extra == True:
			print(col_name)
			print(val_counts)
			print(zero_counts)
			print(non_zero_counts)
			print("TOTAL SUM",sum_val_counts)
			print("NON-ZERO COUNTS",sum_nonzero_counts)
			print("ZERO COUNTS",num_zero_counts)
			print("\n{} is {:2.2f} % non-zero \n\n".format(col_name,percent_non_zero))
	print("AFTER NON ZERO DROPS",df.shape)
	return non_zero_lst


# RDKit 2d

# RDKit 3d

# Morgan FP
def _gen_morgan_fp(smiles_list, fp_radius: int=3, num_bits: int=32768, 
				display_smi_num: int=None, num_bits_display: int=None):
	#mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
	mol_list = []
	for smi in smiles_list:
		if smi is np.nan:
			mol_list.append(np.nan)
		elif type(smi) == float:
			print("Warning: Float detected in Smiles",smi)
		else:
			mol_list.append(Chem.MolFromSmiles(smi))
		
	mfp_list = []
	for mol in mol_list:
		bit_dict = {}
		#print(mol)
		if mol is np.nan:
			mfp = np.nan
		elif mol is not None:
			mfp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=num_bits, radius=fp_radius, bitInfo=bit_dict, useFeatures=True) #added features
		else:
			mfp = np.nan
		if mol_list.index(mol) == display_smi_num:
			#mol, bitId, bitInfo
			# generate tuples list
			bit_tuples_list = []
			for bit_num,values in bit_dict.items():
				bit_tuples_list.append((mol,bit_num,bit_dict))
			#print(*bit_tuples_list,sep='\n')
			multi_bits_img = Draw.DrawMorganBits(bit_tuples_list, useSVG=True)
			#;;display(mol)
			#;;display(multi_bits_img)
		mfp_list.append(mfp)

	return mfp_list

def morgan_fp_from_smi_superlist(smi_superlist: list, label_list: list, 
									fp_rad: int=3, n_fp_bits: int=32768, 
									hush: bool=True):
	#;label_list = ['mono1', 'mono2', 'solvent']
	#mfp_superlist = []
	morgan_df_superlist = []
	for smi_lst,label in zip(smi_superlist,label_list):
		print(f"Processing Morgan FP for {label}...")
		tmp = _gen_morgan_fp(smi_lst, fp_rad, n_fp_bits) 
		#for x in tmp:
		tmp = [np.array(x, dtype=np.intc) for x in tmp] #! This technically doesn't need to be signed
		#mfp_superlist.append(tmp)
		tmp_arr = np.array(tmp)
		if not hush:
			print(f"\tbits in use:", np.logical_or.reduce(tmp_arr).sum())

		tmp_arr_stacked = np.stack([x for x in tmp_arr],axis=0)
		if not hush:
			print("\tStacked shape:",tmp_arr_stacked.shape)
		tmp_col_labels = [f'morgan_{label}_{num}' for num in range(0,tmp_arr_stacked.shape[1])]
		assert len(tmp_col_labels) == tmp_arr_stacked.shape[1]
		tmp_morgan_df = pd.DataFrame(tmp_arr_stacked, columns=tmp_col_labels)

		# Ensure no NaNs are in columns
		print("\tNo NaNs found" if not tmp_morgan_df.isna().any().any() else "FOUND NaNs, BEWARE")
		morgan_df_superlist.append(tmp_morgan_df)

	if len(morgan_df_superlist) == 1:
		print("\tOnly one smiles list given, returning dataframe...")
		df_morgan_out = morgan_df_superlist[0]
		return 
	else:
		#df_morgan_out = pd.concat([df1, df3], sort=False)
		#return df_morgan_out
		for i in range(0, len(morgan_df_superlist) - 1):
			#print(i, i+1)
			print(f"\tJoining dataframes for {label_list[i]} and {label_list[i+1]}")
			if i == 0:
				df_morgan_out = morgan_df_superlist[i].join(morgan_df_superlist[i+1], how='left')
			else:
				df_morgan_out = df_morgan_out.join(morgan_df_superlist[i+1], how='left')
	return df_morgan_out

# RDKit FP
def _gen_rdkit_fp(smiles_list: list, min_path: int=1, max_path: int=7,
				        num_bits: int=32768, n_BitsPerHash: int=2, use_Hs: bool=True,
				        tgt_Density: float=0.0, min_size: int=128):
    rdkit_fp_list = []
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    for m in mol_list:
        #rd_fp = Chem.RDKFingerprint(m, minPath=min_path,  maxPath=max_path,  fpSize=num_bits,  nBitsPerHash=n_BitsPerHash, 
        #		 useHs=use_Hs,  tgtDensity=tgt_Density,  minSize=min_size,  branchedPaths=True,
        #		 useBondOrder=True,  atomInvariants=0,  fromAtoms=0,  atomBits=None, bitInfo=None)
        #Chem.RDKFingerprint(test_mol, 1, 7,
				#  2048, 2, True,
				#  0.0, 128, True, True, 0, 0, None, None)
        rd_fp = Chem.RDKFingerprint(m, min_path,  max_path,  num_bits,
                                     n_BitsPerHash, use_Hs,  tgt_Density,  min_size,  True,
        		                         True,  0,  0,  None, None)
        rd_fp_arr = np.array(rd_fp)
        rdkit_fp_list.append(rd_fp_arr)
    return rdkit_fp_list

def rdkit_fp_from_smi_superlist(smi_superlist: list, label_list: list, 
									fp_rad: int=3, n_fp_bits: int=32768, 
									hush: bool=True):
	#;label_list = ['mono1', 'mono2', 'solvent']
	#mfp_superlist = []
	rdfp_df_superlist = []
	for smi_lst,label in zip(smi_superlist,label_list):
		print(f"Processing rdkit FP for {label}...")
		tmp = _gen_rdkit_fp(smi_lst,
							 min_path=1, max_path=7,
					         num_bits=n_fp_bits, n_BitsPerHash=2, use_Hs=True,
					         tgt_Density=0.0, min_size=128)
		#for x in tmp:
		tmp = [np.array(x, dtype=np.intc) for x in tmp] #! Changed
		#mfp_superlist.append(tmp)
		tmp_arr = np.array(tmp)
		if not hush:
			print(f"\tbits in use:", np.logical_or.reduce(tmp_arr).sum())

		tmp_arr_stacked = np.stack([x for x in tmp_arr], axis=0)
		if not hush:
			print("\tStacked shape:",tmp_arr_stacked.shape)
		tmp_col_labels = [f'rdkit_{label}_{num}' for num in range(0,tmp_arr_stacked.shape[1])]
		assert len(tmp_col_labels) == tmp_arr_stacked.shape[1]
		tmp_rdfp_df = pd.DataFrame(tmp_arr_stacked, columns=tmp_col_labels)

		# Ensure no NaNs are in columns
		print("\tNo NaNs found" if not tmp_rdfp_df.isna().any().any() else "FOUND NaNs, BEWARE")
		rdfp_df_superlist.append(tmp_rdfp_df)

	if len(rdfp_df_superlist) == 1:
		print("\tOnly one smiles list given, returning dataframe...")
		df_rdfp_out = rdfp_df_superlist[0]
		return 
	else:
		#df_rdfp_out = pd.concat([df1, df3], sort=False)
		#return df_rdfp_out
		for i in range(0, len(rdfp_df_superlist) - 1):
			#print(i, i+1)
			print(f"\tJoining dataframes for {label_list[i]} and {label_list[i+1]}")
			if i == 0:
				df_rdfp_out = rdfp_df_superlist[i].join(rdfp_df_superlist[i+1], how='left')
			else:
				df_rdfp_out = df_rdfp_out.join(rdfp_df_superlist[i+1], how='left')
	return df_rdfp_out

# Morfeus

