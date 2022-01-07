# pawpularity-kaggle
Platform for Pawpularity Kaggle contest https://www.kaggle.com/c/petfinder-pawpularity-score  

# Models tried
- Gamma Regression on meta data: RMSE ~20  
- Mixture Model on meta data, distributions adopted are Gamma for pawpularity and Bernoulli for meta data: RMSE ~20
- Pretrained ResNet18 followed by Gamma Regression: RMSE ~18
- Pretrained ResNet18 followed by MLP: RMSE ~18

# Further explorations
- Multitask Learning 
- Ways to use meta data
