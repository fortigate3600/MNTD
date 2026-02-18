# MNTD

This repository provides an implementation of the method for detecting Trojan attacks in machine learning models, you can find the paper [here](https://arxiv.org/abs/1910.03137).

## Installation

Install the required dependencies with:
```bash
pip install -r requirements.txt
```

to generate the shadow models:
```bash
python generate_clean_model.py -file_name model_mnist_lr0.0075 -n 256
```
```bash
python generate_poison_model.py -file_name model_mnist_lr0.0075 -n 256
```
once done train the meta trainer with:
```bash
python meta_trainer.py -file_name model_mnist_lr0.0075 -training query
```
Then you can train a smart attacker that wants to evade the meta classifier:
```bash
python adaptive_attack.py -file_name model_mnist_lr0.0075 -m mntd_query.pt
```
In the end to test the evader against the classifier
```bash
python test_meta-classifier.py -file_name model_mnist_lr0.0075 -trojan_model_path ../shadow_models/adaptive_models/adaptive_model_0.pt -mntd_path mntd_query.pt
```
