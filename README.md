# Domain Generalization for Punch Mark Classification
Code used for the paper "Domain Generalization and Punch Mark Classification" currently under review.

Most of the code in the domainbed folder is from the [DomainBed](https://github.com/facebookresearch/DomainBed/tree/8ee9b5831bc733738361ae119b1f1dd1d29a4ae8?tab=readme-ov-file) repository, associated with the paper [In Search of Lost Domain Generalization](https://openreview.net/pdf?id=lQdXeXDoWtI). Portions of DomainBed have been adapted for the project at hand. The DomainBed version used (8ee9b5831bc733738361ae119b1f1dd1d29a4ae8), with our modifications, is included in this repository to aid reproducibility.

Other CNN baselines are in the CNN_Baseline_Comparisons folder (for comparison with ResNets and DenseNets) and Single_Domain_Comparisons folder (for training on only a single domain). The code has been slightly modified in some instances e.g. to remove extra print statements and modify comments.

## Experiments

### Testing DomainBed DG Methods and Baselines
To test each method in three experiments (of three trials each), we use the following commands:
```sh
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed [seed_number] --trial_seed [trial_number] --output_dir results/test_[algorithm]_S[seed_number]_T[trial_number] --algorithm [algorithm]
```
where \[algorithm\] is replaced by ERM, IRM, GroupDRO, CORAL, Mixup, MLDG, DANN, CDANN, EQRM or SagNet. \[trial_number\] and \[seed_number\] each range betwen 0 and 3 (for a total of 9 runs). The test_env indices can be updated to change train/test domains. The output_dir parameter can be changed to anything as needed. For the ablation study, \[algorithm\] is also replaced by \[ERM_unfreeze\] or \[ERM_unpretrained\] or \[ERM-unfreeze_unpretrained/], and the hparams_registry.py file is replaced by the contents of the hparams_registry-noAug.py (for no data augmentation), or hparams_registry-ResNet18.py (for replacing the ResNet-50 backbone with a ResNet-18) or hparams_registry-ResNet18-noAug.py (for no data augmentation and a ResNet-18). So, a single method might look like:
```sh
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 0 --trial_seed 0 --output_dir results/test_ERM_S0_T0 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 1 --trial_seed 0 --output_dir results/test_ERM_S1_T0 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 2 --trial_seed 0 --output_dir results/test_ERM_S2_T0 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 0 --trial_seed 1 --output_dir results/test_ERM_S0_T1 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 1 --trial_seed 1 --output_dir results/test_ERM_S1_T1 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 2 --trial_seed 1 --output_dir results/test_ERM_S2_T1 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 0 --trial_seed 2 --output_dir results/test_ERM_S0_T2 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 1 --trial_seed 2 --output_dir results/test_ERM_S1_T2 --algorithm ERM
python -m domainbed.scripts.train --dataset PunchMarksDefault --test_env 0 1 2 3  --holdout_fraction 0.2 --seed 2 --trial_seed 2 --output_dir results/test_ERM_S2_T2 --algorithm ERM
```
For these experiments we use the environment given in Environments/dgEnv0.yml.


### Hyperparameter Sweep Tests
To run the random hyperparameter experiments, we use the following:
```sh
python -m domainbed.scripts.sweep launch --data_dir=[dataset_directory] --datasets PunchMarksSweep --command_launcher local --n_hparams 8 --n_trials 3 --output_dir=sweepResults/sweep-[algorithm] --algorithms [algorithm]
```
where again \[algorithm\] is replaced by ERM, IRM, GroupDRO, CORAL, Mixup, MLDG, CDANN, EQRM, DANN, or SagNet. \[dataset_directory\] may need to be changed to the directory where the dataset is located (and/or changed directly in domainbed/datasets.py. For these experiments we use the environment given in Environments/dgEnv1.yml.

### CNN Baselines
For convolutional neural network baselines, we simply run the python files in the CNN_Baseline_Comparisons folder. For instance,
```sh
python baseline-resnet50-no_aug-no_pretraining.py
```
will run the ResNet-50 baseline without data augmentation and without using pretraining. The output directory (and file input directories) should be changed directly in the .py file itself. For these experiments we use the environment given in Environments/dgEnv2.yml.

### Single Training Domain Experiments
Similar to the convolutaional neural network baselines, but with files in the Single_Domain_Comparisons folder. Each baseline convolutional neural network is run twice, depending on the single domain used for training, e.g.
```sh
python baseline-sfa-resnet50-aug.py
python baseline-slf-resnet50-aug.py
```
Would give results when a ResNet-50 (with data augmentaiton) is trained on the sfa or slf domain. Again, filepaths and any parameters should be changed in the .py directly. For these experiments we use the environment given in Environments/dgEnv2.yml.

### Data
We use an torchvision.datasets.ImageFolder directory structure (and using a custom class which inherits the DomainBed MultipleDomainDataset class).

To get the train/validation splits created by DomainBed (so that baseline convolutional neural network baselines have the same train/valid splits), we use the following
```sh
python -m domainbed.scripts.getDomainSplits --dataset PunchMarksDefault --test_env 0 1 2 3  --seed [seed_number] --trial_seed [trial_number] --output_dir dataSplits/dataSplit_default_trial[trial_number]-[seed_number]
```
where as before \[trial_number\] ranges between 0 and 3 (and \[seed_number\] is arbitrary and should not affect the outcome, and so can be set to 0). This will show the data augmentations used. To remove these data augmentations, the datasets.py file should be modified to make the augmentation transform match the default transform. Our training domains come from the paper "[An Artificial Intelligence System for Automatic Recognition of Punches in Fourteenth-Century Panel Painting](https://ieeexplore.ieee.org/document/10016708)" and the images can be found at the associated [repository](https://github.com/marcozullich/punches_recognition).

### Analysis
We compute summary statistics using the scripts found in the analysis folder. These include train-domain validation and test-domain validation model selection for the experiments (when following the folder/name structure used above).

Each script should be run after modifying the data directories as described in each file. Additional details are given in each script, but analysis_3x3.py is used for the paper's main experiment, testing each DomainBed method (or variations on ERM) in three experiments of three trials each; the output of each method should be given its own folder in the root directory. analysis_sweep.py is used to collect results for hyperparameter sweeps and analysis_sweep-385.py is used to track these results for an additional microscopy domain (punch class 385). In each case, all output folders of the sweep should be combined into a single root folder. analysis_cnn_baseline.py (and analysis_cnn_baseline-385.py) are used to track accuracies on the different test domains (and a fourth test domain 385 in the case of  analysis_cnn_baseline-385.py), using the output of the CNN baselines in the CNN_Baseline_Comparisons and Single_Domain_Comparisons folders, again with each CNN given its separate folder in a root directory.

Results can be found in our paper "Domain Generalization and Punch Mark Classification" that is currently under review. If the code in this repository is useful for you, please consider citing this work.

