import pandas as pd
import numpy as np
import pickle
import json
from argparse import ArgumentParser
from datetime import datetime
import sys
from sys import exit
from itertools import chain

import sklearn
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict,cross_validate,KFold

from pathlib import Path


########### ARGUMENT PARSER ###########
parser = ArgumentParser()
parser.add_argument('-d','--descriptors', help='List of descriptors to run. (atombd,mordred,atommordred,mfp,atommfp,rdfp,atomrdfp)',  type=lambda s: [item.strip() for item in s.split(',')], required=True)
parser.add_argument('-m','--models', help='List of ML models to train. e.g. -m "2,3,4,5". 2-DecisionTree 3-RandForest 4-AdaBoost 5-Naive Bayes', type=lambda s: [int(item) for item in s.split(',')], required=True)
parser.add_argument('--debug', help='Flag for whether or not a dry/debugging run should be performed.', type=bool)
parser.add_argument('--seed', help='Flag to set the random seed for data split and model training. Adds to file out name', type=int)
parser.add_argument('--name', help='Name for model used.', type=str)
parser.add_argument('--nprocs', type=int, default=10, help='Number of CPU nodes to use. (default=10)')
args = parser.parse_args()
print(args,"AT TIME OF SETTING")

if args.seed:
    rng = args.seed
else:
    rng = 0


file_suffix = 'nr'


########### IMPORT AND DEFINE DATA ###########
print("Current working directory:",Path.cwd())
data_path = Path.cwd().parent/"data"
csv_path = data_path/"csvs"
json_path = data_path/"jsons"
pkl_path_base = data_path/"pkls"
pkl_path = (data_path)/f"pkls/2D_atactic_NOPE_{file_suffix}_fm"
#descriptor_path = csv_path/"descriptor_csvs"



# Read Data
dfnr = pd.read_pickle(pkl_path_base/f"df_atactic_NOPE_{file_suffix}_norad.pkl") # used to be 'df_diblock_...'
atom_bd_2d_rdkit_norad = pd.read_json(json_path/f"atactic_NOPE_{file_suffix}_atom_bd_2d_rdkit_norad.json")


mordred_norad_2d = pd.read_csv(csv_path/f"atactic_NOPE_{file_suffix}_mordred_descriptors_norad_noempty.csv")
morgan_fp_norad = pd.read_csv(csv_path/f"atactic_NOPE_{file_suffix}_morgan_fp_32768_norad.csv")
rdkit_fp_norad= pd.read_csv(csv_path/f"atactic_NOPE_{file_suffix}_rdkit_fp_32768_norad.csv")
atom_bond_mordred_2d = pd.read_csv(csv_path/f"atactic_NOPE_{file_suffix}_atom_bond_mordred_2d.csv")
atom_bond_morgan_fp_2d = pd.read_csv(csv_path/f"atactic_NOPE_{file_suffix}_atom_bond_morganfp_32768_2d.csv")
atom_bond_rdkit_fp_2d = pd.read_csv(csv_path/f"atactic_NOPE_{file_suffix}_atom_bond_rdkitfp_32768_2d.csv")


""" 
abd - atom bond descriptors
mrd - mordred
mfp - morgan fingerprint
rdfp - rdkit fingerprint
mrph - morfeus
"""

X_abd = atom_bd_2d_rdkit_norad
X_mrd = mordred_norad_2d
X_mfp = morgan_fp_norad
X_rdfp = rdkit_fp_norad
X_abd_mrd = atom_bond_mordred_2d
X_abd_mfp = atom_bond_morgan_fp_2d
X_abd_rdfp = atom_bond_rdkit_fp_2d

      
y = dfnr['solvent_characteristic']

X_abd_train, X_abd_test, y_train, y_test = train_test_split(X_abd, y, test_size=0.25, random_state=rng)
X_mrd_train, X_mrd_test, y_train, y_test = train_test_split(X_mrd, y, test_size=0.25, random_state=rng)
X_mfp_train, X_mfp_test, y_train, y_test = train_test_split(X_mfp, y, test_size=0.25, random_state=rng)
X_rdfp_train, X_rdfp_test, y_train, y_test = train_test_split(X_rdfp, y, test_size=0.25, random_state=rng)
X_abd_mrd_train, X_abd_mrd_test, y_train, y_test = train_test_split(X_abd_mrd, y, test_size=0.25, random_state=rng)
X_abd_mfp_train, X_abd_mfp_test, y_train, y_test = train_test_split(X_abd_mfp, y, test_size=0.25, random_state=rng)
X_abd_rdfp_train, X_abd_rdfp_test, y_train, y_test = train_test_split(X_abd_rdfp, y, test_size=0.25, random_state=rng)


########### MODEL DEFINITION ###########
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Decision Tree",
    "Random Forest",
    "AdaBoost",
    "Naive Bayes",
]


classifiers = [
    KNeighborsClassifier(3),  
    SVC(kernel="linear", C=0.025, random_state=rng), 
    DecisionTreeClassifier(random_state=rng), 
    RandomForestClassifier(n_estimators=100, random_state=rng), 
    AdaBoostClassifier(random_state=rng), 
    GaussianNB(),  
]


names_to_run = []
classifiers_to_run = []
model_numbers_str = (''.join([str(x) for x in args.models]))
for model_num in args.models:
    names_to_run.append(names[model_num])
    classifiers_to_run.append(classifiers[model_num])
if len(names_to_run) == 0:
    sys.exit("Run failed - no model names found.")
    
    
########### DESCRIPTOR SET FLAGS ###########
run_atom_bd = ['atombd' in args.descriptors][0]
run_mordred = ['mordred' in args.descriptors][0]
run_atom_mordred = ['atommordred' in args.descriptors][0]
run_mfp = ['mfp' in args.descriptors][0]
run_atom_mfp = ['atommfp' in args.descriptors][0]
run_rdfp = ['rdfp' in args.descriptors][0]
run_atom_rdfp = ['atomrdfp' in args.descriptors][0]


descrips = '_'.join(args.descriptors)
descriptor_bools = [run_atom_bd, run_mordred, run_atom_mordred, run_mfp, run_atom_mfp, run_rdfp, run_atom_rdfp]



########### DEFINE LOG/LOGGING FUNCTION ###########
log_dir =  Path.cwd().parent/"logs"
exec_time = datetime.now()
year = exec_time.year
month = exec_time.month
day = exec_time.day
hour = exec_time.hour
minute = exec_time.minute
is_pm = hour > (hour % 12)
if is_pm == True: 
    hour_pm = hour - 12
    if minute < 10:
        minute = f"0{minute}"
    hourmin = f"{hour_pm}-{minute}PM"
else:
    hourmin = f"{hour}-{minute}AM"

log_name = f"{year}_{month}_{day}__{hourmin}_HomopolyMLRun_{descrips}.txt"
log_path = log_dir/log_name

def write_to_log(txt: str):
    #txt = '\n' + txt
    with open(log_path, 'a') as f:
        f.write('\n')
        f.write(txt)


########### DEBUG ###########
write_to_log(f'Starting model run at    {exec_time}')
write_to_log(f'Descriptors:    {descrips}')
write_to_log(f'Models:    {names_to_run}')
write_to_log(f'Model Numbers: {model_numbers_str}')
write_to_log(f'Descriptor bools:    {descriptor_bools}')
write_to_log(f'PCA NOT Used for Mordred/MorganFP/RDKitFP Data:    True')
write_to_log(f'Random seed used: {rng}')
write_to_log(f'Num processors to use: {args.nprocs}')
write_to_log(f'Raw Parser args: {args}')

if args.debug == True:
    print(args)
    print(descriptor_bools)
    write_to_log('Debug run exited, log written')
    sys.exit("Debug run exited, log written")
    
    
    
########### MODELS FUNCTION ###########
from joblib import parallel_backend

def gen_ml_models(model_names: list, model_funcs: list, X_train, y_train, X_test, y_test):
    models_list_out = []
    kfold_list_out = []
    for name, clf in zip(model_names, model_funcs):
        model_start_time = datetime.now()
        write_to_log(f'\nSTART {name} at {model_start_time}')
        with parallel_backend('threading', n_jobs=args.nprocs): # NOTE Used to be hardcoded 10
            model_fit = clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        models_list_out.append([name, model_fit, score])
        cv_results = cross_val_score(clf, # Pipeline
                                 X_train, # Feature matrix
                                 y_train, # Target vector
                                 cv=KFold(n_splits=5, shuffle=True, random_state=rng), # Cross-validation technique
                                 scoring="accuracy", # Loss function
                                 n_jobs=args.nprocs) # NOTE Changed to use args.nprocs (default 10).
        kfold_list_out.append([name, model_fit, np.mean(cv_results), score, cv_results])
        model_end_time = datetime.now()
        write_to_log(f'DONE {name} at {model_end_time}')
        write_to_log(f'Run time: {model_end_time - model_start_time}')
        write_to_log(f'\t{name} Results:\n\t  Score: {score:.4f}\n\t  CV Average Score: {np.mean(cv_results):.4f}')

    model_scores = [x[2] for x in models_list_out]
    best_model_accuracy = np.max(model_scores)
    best_model_index = model_scores.index(best_model_accuracy)
    best_model_name = models_list_out[best_model_index][0]
    write_to_log(f'Best model is {best_model_name} with an accuracy of {best_model_accuracy:.4f}')

    return models_list_out, kfold_list_out



        
########### RUN MODELS ###########
if run_atom_bd == False & run_mordred == False & run_atom_mordred == False: # to be changed
    no_descriptors_warning = "Warning: All models appear to be false!"
    print(no_descriptors_warning)
    write_to_log(no_descriptors_warning)    


X_data_tuples = [('Atactic Atom and Bond (No PE)', f'at_NOPE_{file_suffix}_atom_bd_', X_abd_train, X_abd_test),
                    ('Atactic Mordred (No PE)', f'at_NOPE_{file_suffix}_mordred_', X_mrd_train, X_mrd_test),
                    ('Atactic Morgan FP (No PE)', f'at_NOPE_{file_suffix}_mfp_', X_mfp_train, X_mfp_test),
                    ('Atactic RDKit FP (No PE)', f'at_NOPE_{file_suffix}_rdfp_', X_rdfp_train, X_rdfp_test),
                    ('Atactic Atom + Mordred (No PE)', f'at_NOPE_{file_suffix}_atom_mordred_', X_abd_mrd_train, X_abd_mrd_test),
                    ('Atactic Atom + Morgan FP (No PE)', f'at_NOPE_{file_suffix}_atom_morganfp_', X_abd_mfp_train, X_abd_mfp_test),
                    ('Atactic Atom + RD FP (No PE)', f'at_NOPE_{file_suffix}_atom_rdfp_', X_abd_rdfp_train, X_abd_rdfp_test)]


pkl_out_path = pkl_path # Modify as needed.


all_model_scores = []
for i,descrip_bool in enumerate(descriptor_bools):
    if descrip_bool == True:
        long_name, short_name, X_train_data, X_test_data = X_data_tuples[i]
        write_to_log(f"~~~~~~~~\t{long_name.upper()} RUN\t~~~~~~~~")
        curr_models_list, curr_kfold_list = gen_ml_models(names_to_run, classifiers_to_run, X_train_data, y_train, X_test_data, y_test)
        all_model_scores.append([long_name] + [(x[0],x[2]) for x in curr_models_list])
      
        
        model_list_name = short_name + model_numbers_str + "_models_list.pkl"
        kfold_list_name = short_name + model_numbers_str + "_kfold_list.pkl"
        X_train_name = short_name + model_numbers_str + "_X_train.pkl"
        X_test_name = short_name + model_numbers_str + "_X_test.pkl"

        if args.seed is not None:
            model_list_name = short_name + model_numbers_str + "_NOPCA" + f"_rs{args.seed}" "_models_list.pkl"
            kfold_list_name = short_name + model_numbers_str + "_NOPCA" + f"_rs{args.seed}" "_kfold_list.pkl"
            X_train_name = short_name + model_numbers_str + "_NOPCA" + f"_rs{args.seed}" "_X_train.pkl"
            X_test_name = short_name + model_numbers_str + "_NOPCA" + f"_rs{args.seed}" "_X_test.pkl"
        else:
            model_list_name = short_name + model_numbers_str + "_NOPCA" + "_models_list.pkl"
            kfold_list_name = short_name + model_numbers_str + "_NOPCA" + "_kfold_list.pkl"
            X_train_name = short_name + model_numbers_str + "_NOPCA" + "_X_train.pkl"
            X_test_name = short_name + model_numbers_str + "_NOPCA" + "_X_test.pkl"
            
            
        with open(pkl_out_path/model_list_name, 'wb') as f1: # Check path
            pickle.dump(curr_models_list, f1)
        with open(pkl_out_path/kfold_list_name, 'wb') as f2: # Check path
            pickle.dump(curr_kfold_list, f2)
            
        with open(pkl_out_path/X_train_name, 'wb') as f3: # Check path
            pickle.dump(X_train_data, f3)
        with open(pkl_out_path/X_test_name, 'wb') as f4: # Check path
            pickle.dump(X_test_data, f4)
            
        write_to_log(f"FINISHED {long_name.upper()} RUN\n")


write_to_log("Writing y train and test sets...")
# Below should be changed to work with RS (currently overwrites if a different RS is used)
with open(pkl_out_path/(f"{model_numbers_str}_atactic_NOPE_{file_suffix}_y_train_data.pkl"), 'wb') as f5: 
    pickle.dump(y_train, f5)

# Below should be changed to work with RS (currently overwrites if a different RS is used)
with open(pkl_out_path/(f"{model_numbers_str}_atactic_NOPE_{file_suffix}_y_test_data.pkl"), 'wb') as f6: 
    pickle.dump(y_test, f6)
write_to_log("Done writing y data.")


print("ALL MODEL SCORES:", *all_model_scores, sep='\n')
for mod_list in all_model_scores:
    mod_name = mod_list[0]
    log_string = f"{mod_name}"
    for tup in mod_list[1:]:
        log_string = f"{log_string}\n\t{tup}"
    write_to_log(log_string)
write_to_log(f"ALL ML RUNS FINISHED at {datetime.now()}")
