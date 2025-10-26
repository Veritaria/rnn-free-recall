# A neural network model of free recall learns multiple memory strategies

### Getting started
Install required packages:
```
pip install -r requirements.txt
```

To run the model with the default setup:
```
python -u main.py --exp Basic --setup setup.json
```
Wait for the model to complete training and analysis, and check results in /experiments/Basic/figures.

The model shall show the memory palace strategy in free recall.

Check the setup file in /experiments/Basic/setups to get the default training settings. You can then modify them according to your needs.

Python files in /experiments/Basic include analyses of the model. The default analysis file is experiment.py. To change an analysis file to run, add a parameter --exp_file [Your_Exp_File] when running main.py.

### Guidance on replication
- To run a number of models with different hyperparameters, run all setups in /experiments/VaryAllSeq8/setups. A total of 720 models will be trained, which can be modified in the setting files.
- To replicate clustering results in the paper, check analysis_fr_paper.ipynb
- To submit a number of jobs with slurm, check run.sh and run_cluster.py
- To run TCM or the reservoir RNN, check /experiments/TCM and /experiments/Reservoir
- To modify model structure, check /models/model/value_gru.py
- To modify the task, check /tasks/FreeRecall.py

### Description of files
The structure of the repo is as follows.
```
.
├── analysis            # Analyses code of model behavior and representation
│   ├── behavior        # Behavior metrics of free recall
│   ├── decoding        # Decoding analyses of item index and identity in hidden states
│   ├── decomposition   # Includes PCA on hidden states
│   ├── dynamics
│   ├── perturbation
│   └── visualization
├── experiments
│   ├── Basic                       # A simple setup for quick start
│   │   ├── experiment.py           # Default set of analyses on the trained models
│   │   ├── item_invariant.py       # Cross-decoding of item index
│   │   ├── perturbation.py         # Noise robustness analysis
│   │   ├── setups
│   │   │   └── setup.json          # Training configuration file, including model structure, task and training settings
│   │   └── time_invariant.py       # Cross-decoding of item identity
│   ├── CondFR                      # Training the models on conditional free recall task
│   ├── LongSeq                     # Training the models on sequence length of 12 and 16
│   ├── Performance                 # Training a large number of models with the same hyperparameter setting, used to generate figure 4
│   ├── Reservoir                   # Reservoir networks
│   ├── TCM                         # Replication and analyses of TCM
│   ├── VaryAllSeq8                 # A group of models trained with different hyperparameter settings, used for the clustering analysis
│   ├── VaryAllSeq8LargeNoise       # A group of models same as above, but with a larger amount of noise injected to working memory
│   └── VaryAllSeq8NoNoise          # A group of models same as above, but with no noise injected to working memory
├── models
│   ├── base_module.py              # A basic structure of 
│   ├── memory                      # A set of memory modules
│   │   ├── key_value.py            # The key-value memory module
│   │   ├── similarity
│   │   │   ├── lca.py
│   │   │   ├── similarity.py       # The method of computing memory similarity for memory retrieval used in the paper
│   │   │   └── utils.py
│   │   ├── tcm.py
│   │   └── value.py                # The default memory module of the neural network model
│   ├── model
│   │   ├── tcm.py                  # A classic TCM model
│   │   ├── value_ctrnn.py
│   │   └── value_gru.py            # The default model architecture used in the paper
│   ├── module
│   ├── planning
│   └── utils.py
├── tasks
│   ├── base.py
│   ├── ConditionalFreeRecall.py    # The conditional free recall task used in the paper
│   ├── FreeRecall.py               # The default free recall task used in the paper
│   ├── FreeRecall2.py
│   └── wrappers.py
├── train
│   ├── criterions                  # Criterions for training the model
│   ├── record.py                   # Recording a number of trials including the hidden states after training for analysis
│   ├── train.py                    # Code for training the model
│   └── utils.py
├── utils                           # Essential utils functions for running the code, do not delete
├── analysis_fr_paper.ipynb         # The main file to generate plots for the paper
├── analysis_sample.ipynb           # Include some analyses on the conditional free recall task and models with key-value memory module
├── consts.py                       # Some default paths for storing data
├── main.py                         # Where the main training and analysis start
├── run_cluster.py                  # Generate a number of bash files to submit jobs with slurm
├── run.sh                          # Run all setups in /experiments/VaryAllSeq8 with slurm
└── vary_param.py                   # Generate a number of setup files for different hyperparameters
```


### Contact information
For any questions, comments, or suggestions, please reach out to Moufan Li (moufan.li@nyu.edu).
