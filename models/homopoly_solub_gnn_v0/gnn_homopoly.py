import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'

import json
from collections import namedtuple
import numpy as np
import pandas as pd
import nfp
from tensorflow.keras import layers
import rdkit.Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumHBA, CalcNumHBD, CalcTPSA, CalcLabuteASA


import tensorflow_addons as tfa

class CustomPreprocessor(nfp.SmilesPreprocessor):
    def construct_feature_matrices(self, smiles, train=None):
        features = super(CustomPreprocessor, self).construct_feature_matrices(smiles, train)
        return features
    

    output_signature = {'atom_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solute': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solute': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'atom_solv': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'bond_solv': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                        'connectivity_solv': tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
                        'mol_features_solute': tf.TensorSpec(shape=(4,), dtype=tf.float32), #! Change shape as needed
                        'mol_features_solv': tf.TensorSpec(shape=(4,), dtype=tf.float32)} #! Change shape as needed

def atom_features(atom):
    atom_type = namedtuple('Atom', ['totalHs', 'symbol', 'aromatic', 'fc', 'ring_size'])
    return str((atom.GetTotalNumHs(),
                atom.GetSymbol(),
                atom.GetIsAromatic(),
                atom.GetFormalCharge(), # 220829
                nfp.preprocessing.features.get_ring_size(atom, max_size=6)
               ))

def bond_features(bond, flipped=False):
    bond_type = namedtuple('Bond', ['bond_type', 'ring_size', 'symbol_1', 'symbol_2'])

    if not flipped:
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()

    else:
        atom1 = bond.GetEndAtom()
        atom2 = bond.GetBeginAtom()

    return str((bond.GetBondType(),
                nfp.preprocessing.features.get_ring_size(bond, max_size=6),
                atom1.GetSymbol(),
                atom2.GetSymbol()
               ))

def global_features(smiles, row, solute_or_solvent):
    mol = rdkit.Chem.MolFromSmiles(smiles)

    if solute_or_solvent == 'solute':
        return tf.constant([CalcNumHBA(mol),
                         CalcNumHBD(mol), 
                         CalcLabuteASA(mol),
                         CalcTPSA(mol),
                         ])
    else:
        return tf.constant([CalcNumHBA(mol),
                         CalcNumHBD(mol), 
                         CalcLabuteASA(mol),
                         CalcTPSA(mol),
                         ])

def create_tf_dataset(df, preprocessor, sample_weight = 1.0, train=True): 
    for _, row in df.iterrows():
        inputs_solute = preprocessor.construct_feature_matrices(row['can_smiles_solute'], train=train)
        inputs_solvent = preprocessor.construct_feature_matrices(row['can_smiles_solvent'], train=train)
        if not train:
            one_data_sample_w = 1.0
        else:
            try:
                one_data_sample_w = 1.0 
            except: 
                one_data_sample_w = 1.0

        
        yield ({'atom_solute': inputs_solute['atom'],
                'bond_solute': inputs_solute['bond'],
                'connectivity_solute': inputs_solute['connectivity'],
                'atom_solv': inputs_solvent['atom'],
                'bond_solv': inputs_solvent['bond'],
                'connectivity_solv': inputs_solvent['connectivity'],
                'mol_features_solute': global_features(row['can_smiles_solute'], row,'solute'),
                'mol_features_solv': global_features(row['can_smiles_solvent'], row, 'solvent')},
               row['solub_code_to_onehot'], one_data_sample_w)


def message_block(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, dropout = 0.0, surv_prob = 1.0):
    
    atom_state = original_atom_state
    bond_state = original_bond_state
    global_state = original_global_state
    
    global_state_update = layers.GlobalAveragePooling1D()(atom_state)

    global_state_update = layers.Dense(features_dim, activation='relu')(global_state_update)
    global_state_update = layers.Dropout(dropout)(global_state_update)


    global_state_update = layers.Dense(features_dim)(global_state_update)
    global_state_update = layers.Dropout(dropout)(global_state_update)


    global_state = tfa.layers.StochasticDepth(survival_probability = surv_prob)([original_global_state, global_state_update])


    #################
    new_bond_state = nfp.EdgeUpdate(dropout = dropout)([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([original_bond_state, new_bond_state])


    #################
    new_atom_state = nfp.NodeUpdate(dropout = dropout)([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([original_atom_state, new_atom_state])

    
    return atom_state, bond_state, global_state


def message_block_solu_solv_shared(original_atom_state, original_bond_state,
                 original_global_state, connectivity, features_dim, i, Layers):
    
    atom_state_solute, atom_state_solv1, atom_state_solv2 = original_atom_state
    bond_state_solute, bond_state_solv1, bond_state_solv2 = original_bond_state
    global_state_solute, global_state_solv1, global_state_solv2 = original_global_state
    connectivity_solute, connectivity_solv1, connectivity_solv2 = connectivity

    atom_av, global_embed_dense1, global_embed_dense2, global_residcon, nfp_edgeupdate, bond_residcon, nfp_nodeupdate, atom_residcon = Layers[i]

    #solute
    global_state_update = atom_av(atom_state_solute)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solute = global_residcon([global_state_solute, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    bond_state_solute = bond_residcon([bond_state_solute, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solute, bond_state_solute, connectivity_solute, global_state_solute])
    atom_state_solute = atom_residcon([atom_state_solute, new_atom_state])
   
    #solvent 1
    global_state_update = atom_av(atom_state_solv1)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solv1 = global_residcon([global_state_solv1, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solv1, bond_state_solv1, connectivity_solv1, global_state_solv1])
    bond_state_solv1 = bond_residcon([bond_state_solv1, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solv1, bond_state_solv1, connectivity_solv1, global_state_solv1])
    atom_state_solv1 = atom_residcon([atom_state_solv1, new_atom_state])
    
    #solvent 2
    global_state_update = atom_av(atom_state_solv2)
    global_state_update = global_embed_dense1(global_state_update)
    global_state_update = global_embed_dense2(global_state_update)
    global_state_solv2 = global_residcon([global_state_solv2, global_state_update])
    
    new_bond_state = nfp_edgeupdate([atom_state_solv2, bond_state_solv2, connectivity_solv2, global_state_solv2])
    bond_state_solv2 = bond_residcon([bond_state_solv2, new_bond_state])
    
    new_atom_state = nfp_nodeupdate([atom_state_solv2, bond_state_solv2, connectivity_solv2, global_state_solv2])
    atom_state_solv2 = atom_residcon([atom_state_solv2, new_atom_state])

    #Return 
    atom_state =   [atom_state_solute, atom_state_solv1, atom_state_solv2]
    bond_state =   [bond_state_solute, bond_state_solv1, bond_state_solv2]
    global_state = [global_state_solute, global_state_solv1, global_state_solv2]

    return atom_state, bond_state, global_state


def message_block_no_glob(original_atom_state, original_bond_state, connectivity, features_dim, i):
    atom_state = original_atom_state
    bond_state = original_bond_state

    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity])
    bond_state = layers.Add()([original_bond_state, new_bond_state])

    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity])
    atom_state = layers.Add()([original_atom_state, new_atom_state])

    return atom_state, bond_state


