# pawpularity-kaggle
Platform for Pawpularity Kaggle contest https://www.kaggle.com/c/petfinder-pawpularity-score  

# Models tried
- Gamma Regression on meta data: RMSE ~20  
- Mixture Model on meta data, distributions adopted are Gamma for pawpularity and Bernoulli for meta data: RMSE ~20
- Pretrained ResNet18 followed by Gamma Regression: RMSE ~19
- Pretrained ResNet18 followed by MLP: RMSE ~19
- Pretrained ResNext101 followed by MLP: RMSE ~19
- Pretrained ResNext101 followed by Multitask MLP: RMSE ~19 

# Outcomes 
- Multitask improve very little
- Simply Concatenate the meta data improve very little

# Further explorations
- Multitask Learning 
- Ways to use meta data
- EfficientNet
- Swin Transformer
