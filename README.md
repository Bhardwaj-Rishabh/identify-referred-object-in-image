# Identify-referred-object-in-image
This repository provides a solution for end-to-end referring expression comprehension in the scene.

We have implemented on top of the code available in the following repository :

@article{zhou2019a,
title={A Real-time Global Inference Network for One-stage Referring Expression Comprehension.},
author={Zhou, Yiyi and Ji, Rongrong and Luo, Gen and Sun, Xiaoshuai and Su, Jinsong and Ding, Xinghao and Lin, Chiawen and Tian, Qi},
journal={arXiv: Computer Vision and Pattern Recognition},
year={2019}}

## Step 1: Create a Python 3 virtual environment
python3 -m venv env_real

## Step 2: Install the dependencies
pip install requirement.txt

## Step 3: Activate the virtual environment
source env_real/bin/activate

## Step 4: Train your model

Download the dataset from this [link](https://drive.google.com/file/d/1-hUcb_YGMka-4eLHsjivC8F0fDGg2kLv/view?usp=sharing
)


and put in the folder : data/referit/

Then execute the following command

python code_to_train/main_dist.py "referit_try" --ds_to_use='refclef' --bs=16 --nw=4

## Step 4: Download the pre-trained model from this link

https://drive.google.com/file/d/10KEBYtm0pIaz-GQzbzQXWLcY38YcQqHW/view?usp=sharing

and put in the folder : tmp_test/models

## Step 6: Run the pre-trained model on a new image
Store the test image in input folder

Change the value of argument "img_dir": "./input" in configs/ds_info.json file.

python Code/main_dist.py referit_trained_by_Kritika --ds_to_use='refclef' --resume=True --only_test=True --only_val=False
