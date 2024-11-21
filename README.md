# Domain Generalization for Punch Mark Classification
Code used for the paper "Domain Generalization and Punch Mark Classification" currently under review.

Most of the code in this repository is from the [DomainBed](https://github.com/facebookresearch/DomainBed/tree/8ee9b5831bc733738361ae119b1f1dd1d29a4ae8?tab=readme-ov-file) repository, associated with the paper [In Search of Lost Domain Generalization](https://openreview.net/pdf?id=lQdXeXDoWtI). Portions of DomainBed have been adapted for the project at hand. The DomainBed version used, as well as our modifications, are included in this repository to aid reproducibility. This modified code is in the domainbed folder.

Convolutional Neural Networks (CNNs) used for comparisons and baselines can found in the CNN_comparisons folder.

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

## Data
We use an PyTorch ImageFolder dataset directory structure (inheriting the 


To get the train/validation splits created by DomainBed (so that baseline convolutional neural network baselines have the same train/valid splits), we use the following
```sh
python -m domainbed.scripts.getDomainSplits --dataset PunchMarksDefault --test_env 0 1 2 3  --seed [seed_number] --trial_seed [trial_number] --output_dir dataSplits/dataSplit_default_trial[trial_number]-[seed_number]
```
where as before \[trial_number\] ranges between 0 and 3 (and \[seed_number\] is arbitrary and should not affect the outcome, and so can be set to 0).





