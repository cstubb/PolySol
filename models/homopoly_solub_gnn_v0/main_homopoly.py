import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    device = "/gpu:0"
else:
    device = "/cpu:0"

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import numpy as np
import pandas as pd
from tensorflow.keras import layers
from gnn_homopoly import *
import nfp
import json 
import sys

from argparse import ArgumentParser
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold, train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


def main(args):
   
    suffix = "NOPE_nr"
    data_path = r"/home/cstubbs/projects/solubility/data/"
    csv_path = data_path + "csvs/"
    pkl_path = data_path + "pkls/"
    at_pkl_path = pkl_path + f"2D_atactic_{suffix}_fm/" # NOTE the _fm!
    
     
    data = pd.read_pickle(at_pkl_path + f"df_atactic_{suffix}_norad.pkl")
    data = data.reset_index() #! Necessary for later code, shouldn't impact index assignments at all. 
    

    data.columns = ['index', 'polymer', 'mono_name', 'can_smiles_solute', 'mono_mol', 'solvent', 'can_smiles_solvent', 'solvent_mol', 'solub_code']
    solub_code_to_onehot = to_categorical(list(data['solub_code']), num_classes=2) 
    solub_code_to_onehot = [x for x in solub_code_to_onehot]
    data['solub_code_to_onehot'] = solub_code_to_onehot

    #* Shuffle data order (normally run it w/ order 1.0). 
    data = data.sample(frac=args.data_frac, random_state=1) 
 
    # Separates training+validation from test
    index_train_valid, index_test, dummy_train_valid, dummy_test = train_test_split(data['index'], data['index'], 
                                                                                    test_size = 0.1, random_state = args.random_seed) 
    test_exp = data[data['index'].isin(index_test)]
    train_valid = data[data['index'].isin(index_train_valid)] # BOTH training and validation set

    # Kfold is used to control validation set size/specific indicies
    kfold = KFold(n_splits = 5, shuffle = True, random_state = args.random_seed) # Split training and validation into 5
    train_valid_split = list(kfold.split(train_valid))[args.fold_number] 
    train_index, valid_index = train_valid_split
    train_exp = train_valid.iloc[train_index]
    valid_exp = train_valid.iloc[valid_index]

    train = train_exp
    valid = valid_exp
    test = test_exp




    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(len(train_exp), len(train) - len(train_exp), len(train))
    print(len(valid_exp), len(valid) - len(valid_exp), len(valid))
    print(len(test_exp), len(test) - len(test_exp), len(test))
    total_num_datapoints = len(train_exp) + len(test_exp) + len(valid_exp)
    print("Total num datapoints:",total_num_datapoints)
    ratio_train = 100*(len(train_exp)/total_num_datapoints)
    ratio_valid = 100*(len(valid_exp)/total_num_datapoints)
    ratio_test = 100*(len(test_exp)/total_num_datapoints)

    print(f"Train/Valid/Test: {ratio_train:.1f}/{ratio_valid:.1f}/{ratio_test:.1f}")
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    # Set labels - after training, concatenate train/valid/test
    train['Train/Valid/Test'] = 'Train'
    valid['Train/Valid/Test'] = 'Valid'
    test['Train/Valid/Test'] = 'Test'

    #* Beginning of actual GNN Part
    #* Preprocessor in GNN.py - converts SMILES into Atom and Bond Feature vectors
    #* output_signature: describes shape of input vectors
    #* tf.TensorSpec: dummy placeholder to specify shape (if shape is None, can be variable). 
    #*      If array given doesnt have specificed shape (if shape not None), error is thrown
    #* Construct_feature_matrices: from NFP package (by peter st. john and YJ) https://github.com/NREL/nfp
    #*      Converts SMILES to mol, gets atoms and bonds, and gets specific features requested for each.
    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features,
    )

    ####


    print(f"Atom classes before: {preprocessor.atom_classes} (includes 'none' and 'missing' classes)")
    print(f"Bond classes before: {preprocessor.bond_classes} (includes 'none' and 'missing' classes)")


    train_all_smiles = list( set(list(train['can_smiles_solvent']) + list(train['can_smiles_solute']) ) ) #! CHANGED: 
    #* Initially preprocessor has no info about atom and bond types, so we iterate over all SMILES to get atom and bond classes
    #* Also have bond_tokenizer - shortening/classifying atom and bond feature info. Class #1/2/x will be converted to 64dim vector
    for smiles in train_all_smiles:
        preprocessor.construct_feature_matrices(smiles, train=True)

    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(2,), dtype=tf.int32),
                        tf.TensorSpec(shape=(), dtype=tf.float32)) 
    
    print(f'Atom classes after: {preprocessor.atom_classes}')
    print(f'Bond classes after: {preprocessor.bond_classes}')

    #* Generates input data (incl. all atom,bond,global features defined in preprocessor)
    train_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, True), output_signature=output_signature)\
        .cache().shuffle(buffer_size=1000)\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    valid_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(valid, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    test_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(test, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    ##################
    #* Beginning of ACTUAL GNN OPERATORS
    features_dim = args.num_hidden
    num_messages = args.layers

    #! Define input for solute/solvent
    #solute
    #* Layers.input is a placeholder to receive dict w/ atom_feature_matrix, bond_feature_matrix, connectivity, and global features
    #* In graph: connectivity gives source atom and target atom
    atom_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='atom_solute')
    bond_Input_solute = layers.Input(shape=[None], dtype=tf.int32, name='bond_solute')
    connectivity_Input_solute = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solute')
    global_Input_solute = layers.Input(shape=[4], dtype=tf.float32, name='mol_features_solute') #! Change shape as needed to fit global features

    #solvent
    atom_Input_solv = layers.Input(shape=[None], dtype=tf.int32, name='atom_solv')
    bond_Input_solv = layers.Input(shape=[None], dtype=tf.int32, name='bond_solv')
    connectivity_Input_solv = layers.Input(shape=[None, 2], dtype=tf.int32, name='connectivity_solv')
    global_Input_solv = layers.Input(shape=[4], dtype=tf.float32, name='mol_features_solv') #! Change shape as needed to fit global features
    ######

    #! Define enbedding and dense layers for solute/solvent
    #* Take a group of vectors or strings, convert into a 64/128 dim vector (embed)
    #* Global feature: already numbers, so we only use a 1 dense layer
    # Solute
    atom_state_solute = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solute', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solute)
    bond_state_solute = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solute', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solute)
    global_state_solute = layers.Dense(features_dim, activation='relu')(global_Input_solute)

    # Solvent
    atom_state_solv = layers.Embedding(preprocessor.atom_classes, features_dim,
                                  name='atom_embedding_solv', mask_zero=True,
                                  embeddings_regularizer='l2')(atom_Input_solv)
    bond_state_solv = layers.Embedding(preprocessor.bond_classes, features_dim,
                                  name='bond_embedding_solv', mask_zero=True,
                                  embeddings_regularizer='l2')(bond_Input_solv)
    global_state_solv = layers.Dense(features_dim, activation='relu')(global_Input_solv)
   
    #* Message blocks for solute and solvent independently
    #* If curious about layers, go to GNN.py and look at message_block function
    for i in range(num_messages):
        if args.surv_prob != 1.0:
            surv_prob_i = 1.0 - (((1.0 - args.surv_prob) / (num_messages - 1)) * i)
        else:
            surv_prob_i = 1.0

        atom_state_solute, bond_state_solute, global_state_solute = message_block(atom_state_solute, 
                                                                                   bond_state_solute, 
                                                                                   global_state_solute, 
                                                                                   connectivity_Input_solute, 
                                                                                   features_dim, i, 1.0e-10, surv_prob_i)
        atom_state_solv,   bond_state_solv,   global_state_solv   = message_block(atom_state_solv, 
                                                                                bond_state_solv, 
                                                                                global_state_solv, 
                                                                                connectivity_Input_solv, 
                                                                                features_dim, i, 1.0e-10, surv_prob_i)

    readout_vector = tf.concat([global_state_solute, global_state_solv], -1)

    readout_vector = layers.Dense(features_dim*2, activation='relu')(readout_vector)
    readout_vector = layers.Dropout(args.dropout)(readout_vector)
    readout_vector = layers.Dense(features_dim*2, activation='relu')(readout_vector)
    readout_vector = layers.Dropout(args.dropout)(readout_vector)
    
    prediction = layers.Dense(2, activation='softmax')(readout_vector)

    input_tensors = [atom_Input_solute, bond_Input_solute, connectivity_Input_solute, 
                     atom_Input_solv, bond_Input_solv, connectivity_Input_solv,
                     global_Input_solute, global_Input_solv]

    model = tf.keras.Model(input_tensors, [prediction])



    model.summary()
    ###############################

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(args.lr))
    model_path = "model_files/" + args.modelname + "/best_model.h5"

    checkpoint = ModelCheckpoint(model_path, monitor="val_loss",\
                                 verbose=2, save_best_only = True, mode='auto', period=1 )

    hist = model.fit(train_data,
                     validation_data=valid_data,
                     epochs=args.epoch,
                     verbose=2, callbacks = [checkpoint])


    model.load_weights(model_path)

    train_data_final = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(train, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=args.batchsize)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    train_results = model.predict(train_data_final).squeeze()
    valid_results = model.predict(valid_data).squeeze()
    test_results = model.predict(test_data).squeeze()

   
    train_labels = tf.argmax(tf.convert_to_tensor(list(train['solub_code_to_onehot']), dtype=tf.float32), 1) 
    valid_labels = tf.argmax(tf.convert_to_tensor(list(valid['solub_code_to_onehot']), dtype=tf.float32), 1) 
    test_labels = tf.argmax(tf.convert_to_tensor(list(test['solub_code_to_onehot']), dtype=tf.float32), 1)

    train_labels_pred = tf.argmax(train_results, 1) 
    valid_labels_pred = tf.argmax(valid_results, 1)
    test_labels_pred =  tf.argmax(test_results, 1)

    correct_prediction_train = tf.equal( train_labels_pred,  train_labels)
    correct_prediction_valid = tf.equal( valid_labels_pred,  valid_labels)
    correct_prediction_test = tf.equal(  test_labels_pred,   test_labels )

    acc_train = tf.reduce_mean(tf.cast(correct_prediction_train, "float")).numpy()
    acc_valid = tf.reduce_mean(tf.cast(correct_prediction_valid, "float")).numpy()
    acc_test =  tf.reduce_mean(tf.cast(correct_prediction_test, "float")).numpy()

    train['predicted_prob'] = list(train_results)
    valid['predicted_prob'] = list(valid_results)
    test['predicted_prob'] = list(test_results)

    train['predicted'] = train_labels_pred
    valid['predicted'] = valid_labels_pred
    test['predicted'] =  test_labels_pred

    print("Fold number", args.fold_number)
    print(len(train),len(valid),len(test))
    print("Train:  {}\nValidation:  {}\nTest:  {}".format(acc_train, acc_valid, acc_test))


    pd.concat([train, valid, test], ignore_index=True).to_csv('model_files/' + args.modelname +'/kfold_'+str(args.fold_number)+'.csv',index=False)
    preprocessor.to_json("model_files/"+ args.modelname  +"/preprocessor.json")


def predict_df(df,args):
    # Need to ensure input df is same format as df above
    model = tf.keras.models.load_model('model_files/'+ args.modelname +'/best_model.h5', custom_objects = nfp.custom_objects)
    preprocessor = CustomPreprocessor(
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
    preprocessor.from_json('model_files/' + args.modelname +'/preprocessor.json')
    
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(2,), dtype=tf.float32), 
                        tf.TensorSpec(shape=(), dtype=tf.float32)) 

    df_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(df, preprocessor, args.sample_weight, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=len(df))\
        .prefetch(tf.data.experimental.AUTOTUNE)

    pred_results = model.predict(df_data)
    print("PRED RESULTS",pred_results,pred_results.shape)

    pred_res_insoluble, pred_res_soluble = zip(*pred_results)
    df['predicted_insoluble'] = pred_res_insoluble
    df['predicted_soluble'] = pred_res_soluble
    return df

    ##################

if __name__ == '__main__':
    with tf.device(device):
        parser = ArgumentParser()
        parser.add_argument('-lr', type=float, default=1.0e-4, help='Learning rate (default=1.0e-4)')
        parser.add_argument('-batchsize', type=int, default=1024, help='batch_size (default=1024)')
        parser.add_argument('-epoch', type=int, default=1000, help='epoch (default=1000)')
        parser.add_argument('-layers', type=int, default=5, help='number of gnn layers (default=5)')
        parser.add_argument('-num_hidden', type=int, default=128, help='number of nodes in hidden layers (default=128)')

        parser.add_argument('-random_seed', type=int, default=1, help='random seed number used when splitting the dataset (default=1)')
        parser.add_argument('-split_option', type=int, default=2, help='8:1:1 split options - 0: just a random 8:1:1 split,\
                                                                                              1: Training set: Tier1,2,3, validation/test set: Tier 1 only,\
                                                                                              2: split from Tier 1 + split from Tier 2,3  (default=2)')

        parser.add_argument('-sample_weight', type=float, default=1.0, help='whether to use sample weights (default=0.6) If 1.0 -> no sample weights, if < 1.0 -> sample weights to Tier 2,3 methods')
        parser.add_argument('-fold_number', type=int, default=0, help='fold number for Kfold')
        parser.add_argument('-modelname', type=str, default='test_model', help='model name (default=test_model)')
        parser.add_argument('-data_frac', type=float, default=1.0, help='default=1.0')

        ########
        parser.add_argument('-predict_df', type=str, default='', help='If specified, prediction is carried out for title.csv.csv (default=False)')

        parser.add_argument('-dropout', type=float, default=0.0, help='default=0.0')
        parser.add_argument('-surv_prob', type=float, default=1.0, help='default=1.0')
        args = parser.parse_args()

    if args.predict_df:
        df = pd.read_csv(f'{args.predict_df}.csv')


        df2 = predict_df(df, args) 
        df2.to_csv('prediction_results_{args.predict_df}.csv', index=False)
    else:
        import datetime
        start = datetime.datetime.now()
        main(args)
        end = datetime.datetime.now()
        print(end-start)
