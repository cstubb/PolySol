# PolySol Data Repository
[![DOI](https://zenodo.org/badge/853452569.svg)](https://doi.org/10.5281/zenodo.14376748)

This is a data repository associated with the manuscript titled "Predicting homopolymer and copolymer solubility through machine learning" by Christopher D. Stubbs, Yeonjoon Kim, Ethan C. Quinn, Raúl Pérez-Soto, Eugene Y.-X. Chen, and Seonah Kim.

This repository consists of a few parts:

- 2 polymer solubility databases (homopolymer + copolymer)
- Code to train and analyze ~38 polymer solubility models (classical + GNN for homopolymers/copolymers)
- Code to perform Shapley Additive Value (SHAP) analysis on the homopolymer model
- Code to identify selective solvents for homopolymer additive removal

## Contents
This repository includes:

- **Two new polymer solubility databases** (curated by hand from J. Brandrup's Polymer Handbook).
  - *Homopolymer solubility (1818 datapoints)*
    - File: data/pkls/df_atactic_NOPE_nr_norad.pkl
    - File: data/csvs/df_atactic_NOPE_nr_norad.csv
  - *Copolymer solubility (270 datapoints)*
    - File: data/pkls/df_dicopoly_norad.pkl
    - File: data/csvs/df_dicopoly_norad.csv
- **Multiple new ML models of polymer solubility** (using molecular and/or fingerprint descriptors)
  - *Homopolymer classical*
    - File: scripts/train_homopoly_classical.py
  - *Copolymer classical*
    - File: scripts/train_copoly_classical.py
  - *Homopolymer GNN*
    - File: model_files/main_homopoly.py
  - *Copolymer GNN*
    - File: model_files/main_copoly.py
- **Code to generate ML descriptors for classical models**
  - *Homopolymer*
    - File: scripts/descriptor_gen_homopoly_classical.ipynb
  - *Copolymer*
    - File: scripts/descriptor_gen_copoly_classical.ipynb
- **Code for analyzing model performance via Shapley Additive Explanations (SHAP)**
  - File: scripts/analysis_and_SHAP.ipynb
- **Code to predict selective solvents for plastic additive removal from polyethylene and polystyrene**
  - File: scripts/additive_removal.ipynb

## Using this Repository
To use the database and models in this repository, you will need a working installation of Python (v3.8-3.10) on your computer alongside the required packages (see "Packages Required"). All code was tested in Windows 10 64-bit and CentOS Stream 8, and so it should work on most modern operating systems. Please report any issues with using this code on GitHub.

### Using the PolySolDB Databases

- To use the homopolymer and copolymer solubility databased (called "PolySolDB" collectively), use the pandas `read_pickle` method to load the database pickle files (located in data/pkls/).

```python
import pandas as pd
df_homopoly = pd.read_pickle("data/pkls/df_atactic_NOPE_nr_norad.pkl")
df_copoly = pd.read_pickle("data/pkls/df_dicopoly_norad.pkl)
print("# Datapoints PolySolDB Homopolymer:", df_homopoly.shape[0])
print("# Datapoints PolySolDB Copolymer:", df_copoly.shape[0])
```

### Training Models

- All model training requires a working Python environment, with GPU access and a CUDA setup ideal but not necessary (see "Packages Required" and "Using this Repository"). Getting CUDA and TensorFlow to work together on a GPU can be challenging, so the GNN model code falls back to a CPU if a GPU cannot be found.
- For the classical ML models, descriptor generation is required *before* training - see scripts/descriptor_gen_..._classical.ipynb
  - Generating these descriptors can be quite resource intensive (30-60 minutes), and 16+ GB of RAM with at least 10GB of storage space is recommended to run the code.
- For the GNN models, descriptor generation is included as part of model training.

#### Training Classical Models

- As previously stated, generate descriptors first by running the descriptor_gen jupyter notebooks found in the 'scripts' folder.
- To train the homopolymer model, call the relevant training script as follows (replace the model name with one of your choosing)
  - Additional parameters can be seen in the source code or by using the --help flag (e.g. python scripts/train_homopoly_classical.py)
  - Homopolymer: `python train_homopoly_classical.py -d='atombd,mordred,atommordred,mfp,atommfp,rdfp,atomrdfp' -m='2,3,4,5' --seed=0 --nprocs=10`
  - Copolymer: `python train_copoly_classical.py -d='atombd,mordred,atommordred,mfp,atommfp,rdfp,atomrdfp' --seed=0 --nprocs=10`
- To change the specific descriptors used for training, change the -d flag. Warning: training with less than the 7 descriptor sets may break the modelGroup parsing code.
- *Homopolymer models only*: To change the specific architectures used for training, change the -m flag. Warning: training with more/less than the 4 architectures specified may break the modelGroup parsing code.
- Trained classical models can be found in pkls/2D_atactic_NOPE_nr_fm (homopolymer) or pkls/2D_copoly.
- Use the modelGroup defined in scripts/metrics_gen.py to load classical model results.

#### Training GNN Models

- To train GNN models, first check whether your machine has CUDA and TensorFlow GPU support setup. This is often a machine-specific process, and depends on your graphics card, its supported CUDA versions, the CUDA versions installed, and the TensorFlow version installed (among other factors)
- GPU use is *not* required for GNN model training, but significant slowdowns may occur if a GPU is not used
- To train GNN models, use the following code snippets as an example (other options available by using the --help flag or checking source code)
  - Homopolymer: `python models/homopoly_solub_gnn_v0/main_homopoly.py -n "ExampleHomopolymerModel"`
  - Copolymer: `python models/copoly_solub_gnn_v0/main_copoly.py -n "ExampleCopolymerModel"`
- Trained GNN models will be saved in models/.../model_files. Each folder has the preprocessor used, the best model (best_model.h5), and the prediction results (kfold_#.csv)

### Loading Models

- Classical models
  - Classical models can be loaded using the modelGroup class found in scripts/metrics_gen.py
  - Example usage of the modelGroup class can be found in scripts/analysis_and_SHAP.ipynb
- GNN Models
  - Pre-trained GNN models can be loaded from the .h5 file found in models/.../model_files/.../best_model.h5. To load, you will need to import the nfp package and pass nfp.custom_objects from nfp as custom_objects to the model load call. Rough example code is below.
  - Model results can be found in the same directory as the h5 file, in the csv file named `kfold_?.csv`, where ? is the fold number for that run (0-4, e.g. kfold_0.csv).

```python
def predict_df(df, model_name, csv_file_dir):
    model_dir = Path.cwd()/(f'model_files/{model_name}')
    csv_name = Path(csv_file_dir).stem
    
    model = tf.keras.models.load_model(model_dir/'best_model.h5', custom_objects = nfp.custom_objects)
    preprocessor = CustomPreprocessor( 
        explicit_hs=False,
        atom_features=atom_features,
        bond_features=bond_features)
    preprocessor.from_json(model_dir/'preprocessor.json')
    
    output_signature = (preprocessor.output_signature,
                        tf.TensorSpec(shape=(2,), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32))

    df_data = tf.data.Dataset.from_generator(
        lambda: create_tf_dataset(df, preprocessor, 1.0, False), output_signature=output_signature)\
        .cache()\
        .padded_batch(batch_size=len(df))\
        .prefetch(tf.data.experimental.AUTOTUNE)

    pred_results = model.predict(df_data).squeeze()
    df['predicted'] = pred_results

```

### Model and SHAP Analysis

- All classical model analysis can be found in scripts/analysis_and_SHAP.ipynb, including accuracy/precision/recall and other classification metrics. Additionally, SHAP analysis for the best homopolymer model can be found directly below the model analysis - we use train data for SHAP predictions, but SHAP predictions for test data are also included for comparison.

### Plastic Additive Removal

- All code for plastic additive removal can be found in scripts/additive_removal.ipynb. This code allows us to use polymer solubility model alongside a small molecule solubility model to predict selective solvents for plastic additive removal via dissolution-precipitation.

## Packages Required
All of the following were retrieved from PyPI, but should also be available on conda-forge.  Most model development was done in Python 3.8.13, but should work fine for Python 3.8 - 3.10 (3.7 may also work, but hasn't been tested). Note that a few packages require specific version numbers (nfp, TensorFlow, pandas, RDKit). Other packages have their version specified for reproducibility, and it is recommended to use the versions specified when possible.

#### Utility

- matplotlib (v3.5.3)
- seaborn (v0.12.0)
- JupyterLab (v3.4.5)

#### Descriptor Generation

- mordred (v1.2.0)
- RDKit (v2022.3.5)

#### ML/Vector Math

- numpy (v1.23.2)
- scipy (v1.9.0)
- pandas (v1.4.3)
- scikit-learn (v1.1.2)
- tensorflow (v2.9.1)
- tensorflow-addons (v0.18.0)
- Keras (v2.9.0)
- nfp (v0.3.0 exactly)
- SHAP (v0.41.0 - requires some modifications - see analysis_and_SHAP.ipynb)

## Filing Issues
Please report all issues or errors with code on GitHub wherever possible.
