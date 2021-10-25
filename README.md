# Credit_Risk_Analysis
## Overview of the analysis.
The purpose of this analysis was to apply machine learning to solve a real-world challenge: credit card risk. we used Credit card credit dataset from LendingClub, a peer-to-peer lending services company.  
we created a supervised machine learning model that could explicitly predict credit card risk.
we employed different techniques such as :

1. **imbalanced-learn** and **scikit-learn** libraries to build and evaluate the models using resampling.
2. **RandomOverSampler** and **SMOTE** algorithms to oversample the data .
3. A combinatorial approach of over- and undersampling using the **SMOTEENN** algorithm.
4. **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**,two machine learning models 
   than reduce bias and predict credit risk.

## Results
### Naive Random Oversampling
The balanced accuracy is 64%       
According to the classification report:       
The high_risk precision is about 1% with 61% recall which makes a F1 of 2% only.       
The low_risk population number is high which makes a precision of 100% with 
a recall of 67%.

### SMOTE Oversampling

Got almost the same result as the previous model
The balanced accuracy of 62%       
The high_risk: precision of 1% with a recall of 60%, f1 of 2%      
The low_risk: precision of 100% with a recall of 64%

### ClusterCentroids resampler

The balanced accuracy of 51%          
The high_risk: precision of 1% with a recall of 59%, f1 of 1% only.        
The low_risk: precision of 100% with a recall of 43% due to the high number of false positives

### SMOTEENN model

The balanced accuracy score of 62%      
The high_risk: precision of 1% with a recall of 69%, f1 of 2%         
The low_risk: precision of 100% with a recall of 55% due to the high number of false positives

### BalancedRandomForestClassifier model

The balanced accuracy of 79%          
The high_risk: precision of 4% with a recall of 67%, f1 of 7%          
The low_risk: precision of 100% with a recall of 91% due to the low number of false positives

### EasyEnsembleClassifier model 

The balanced accuracy of 93%       
The high_risk: precision of 7% with a recall of 91%, f1 increase to 14%       
The low_risk: precision of 100% with a recall of 94% due to the low number of false positives

## Summary
The objective was to find the most effective model to predict credit card risk, especially model that can detect high-risk credits. Suffice to say that all of our models leave a large portion of high-risk credits undetectable, except the **EasyEnsembleClassify** model that can detect almost all high-risk credits with a recall of 91%.
