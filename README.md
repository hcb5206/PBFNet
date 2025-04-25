# Project Name
The implementation code of PBFNet and PBFNet-U.

## Installation
Use the following command to install the required dependencies:
```bash
### Install using Conda environment
pip install -r requirements.txt
```
## Data
Before running the program, download the training data, test data, and model files required for each 
task [here](https://drive.google.com/drive/folders/1hi1Vyg6Iwg4aBrGApyO38xUh8s767sTx?usp=drive_link). 
The folder name for each data is the same as the folder name in the program, so please do not change it. 
And make sure the file paths are consistent.

## MFF
### Testing
by executing the command:
```bash
python model_test_nolabel.py
```
The trained PBFNet model can be used for testing.
### Training
To retrain PBFNet, do:
```bash
python main.py
```
## MEF
### Testing
To get label-based test results first, run:
```bash
python model_test.py
```
To get unreferenced test results first, run:
```bash
python model_test_nolabel.py
```
### Training
To retrain PBFNet, do:
```bash
python main.py
```
### Semantic segmentation
To get the semantic segmentation results on the MEF task, run the predict.py file in the DeepLabV3Plus folder:
```bash
python predict.py
```
## VIF
### Testing
To get the test results, run it directly:
```bash
python model_test.py
```
### Training
To retrain PBFNet, do:
```bash
python main.py
```
### Target detection
To get quantitative results for your target assay, run:
```bash
python OB2.py
```
To get a visualization of the target detection, run:
```bash
python object_dectect.py
```
## MMF
### Testing
To get test results for the MRI-PET and MRI-SPECT tasks, run them separately:
```bash
python model_test_PET.py
```
```bash
python model_test_SPECT.py
```
### Training
To retrain PBFNet on the MRI-PET and MRI-SPECT tasks, run them separately:
```bash
python main_PET.py
```
```bash
python main_SPECT.py
```
## Please note:
Before running all programs, please make sure that you have downloaded the relevant data and saved it under 
the correct path. There are various parameter options set up in the program, I have set them up beforehand, 
if you have any changes, please modify them yourself. Test results on different devices may vary slightly.

