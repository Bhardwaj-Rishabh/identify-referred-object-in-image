# Identify-referred-object-in-image
This repository provides a solution for end-to-end referring expression comprehension in the scene

## Step 1: Create a Python 3 virtual environment
python3 -m venv env_real

## Step 2: Install the dependencies
pip install requirement.txt

## Step 3: Download the pre-trained model from this link
https://drive.google.com/file/d/10KEBYtm0pIaz-GQzbzQXWLcY38YcQqHW/view?usp=sharing

## Step 4: Activate the virtual environment
source env_real/bin/activate

## Step 5: Run the pre-trained model on a new image
python Code/main_dist.py referit_trained_by_Kritika --ds_to_use='refclef' --resume=True --only_test=True --only_val=False
