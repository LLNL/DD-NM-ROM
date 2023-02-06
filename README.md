# DD-NM-ROM

## Requirements
-To use the code in this repo, first create an Open-CE v1.7.2 with CUDA-11.4 for Lassen environment following the instructions [here](https://lc.llnl.gov/confluence/display/LC/2022/10/20/Open-CE+v1.7.2+with+CUDA-11.4+for+Lassen).  
-I also recommend creating a virtual environment containing PyTorch that can be run on a CPU machine (e.g. quartz).
For generating POD bases for the problem sizes considered in this repo, the SVD routine is much faster on CPUs. 

## Generating snapshot data for training
To generate the training data, ssh into lassen and run generate_training_data.sh

## Training a DD-LS-ROM

## Training a DD-NM-ROM
- Make sure the driver_LSROM notebook (see previous section) is run on the desired configuration using a CPU to store POD basis data. 
- 
