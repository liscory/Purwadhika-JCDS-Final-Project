# FINAL PROJECT - PROPERTY VALUE ESTIMATION USING REGRESSION MODEL

by: PURWADHIKA JCDS 1202 - MATPLOTLIB TEAM 
- Lis Cory
- Rezki Fauziansyah
- Teuku Muhammad Kemal Isfan

Dataset : DC_Properties.csv 

Source : <a href="https://www.kaggle.com/christophercorrea/dc-residential-properties?select=DC_Properties.csv">Kaggle</a>

---

# Background 
In this project, we position ourselves as a part of the Data Scientist Team in a Financial Institution, MPL Bank, in Washington DC, USA. 
We are assigned to work on a project to develop a Machine Learning (ML) solution. The project owner is the Underwriter Team [1] of MPL Bank. 
We will help the Underwriter Team to make an improvement in their process of underwriting, specifically in the process of property appraisal[2] and valuation.

MPL Bank orders the appraisal through a third party, an appraisal management company [3], in order to comply with the federal appraiser independence requirements[4]. 
However, the appraisal process performed by an external party has a risk of fraud[5] or producing erroneous results[6]. 
Thus, the project owner wants to address these issues.

##### Notes :
For details please refer to 
<a href="https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/Lead%20-%20Residential.ipynb">Lead - Residential.jpynb</a>, 
<a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/Appendix%201.ipynb'>Appendix 1.jpynb</a> 
and 
<a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/Appendix%202.ipynb'>Appendix 2.jpynb</a>.

---

# I. Problem Identification

## 1. Problem Definition

Based on the elicitation process with the project owner, we found that they want to improve the accuracy of their underwriting process, specifically in the process of evaluating the appraisal. 
In the process of evaluating appraisals, there are some risks that the project owner wants to minimize, such as fraud and erroneous appraisal results given by the AMC. 
In addition, there is also a problem that often happens regarding the difference between the agreed offer made by a borrower and the property seller and the actual property valuation. 
Since lenders can’t lend out money more than a property is worth[7], all of these risks may cause the project owner to determine wrong appraisal value and to make a wrong decision whether to give the loan to a borrower[8].

To address these risks and improve their business process, the project owner needs a reliable autonomous system that can provide an estimation value that can be used to compare the value given by the AMC.

The expected output of this project is a system that can make an estimation of an accurate and reasonable value (price) for a property based on the aspects of the property by using ML. 
However, due to the limitation of our time budget, we limit the capability of our model in this project to predict an output only for properties with grade lower than Exceptional, 
since Exceptional properties have a price range that is very different than the rest of properties with other grades.

<p float="center ">
<img src="https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Appendix%20Picture/Figure%2001.png" width="430" />
<img src="https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Appendix%20Picture/Figure%2002.png" width="500" />
  
## 2. Business Objectives

The business goals that want to be achieved through this project are as follows:
- To maximize profit by making the right decision to give a loan with an optimal amount.
- To minimize loss and risks of fraud and erroneous valuation[9].

The ML system that will be built need to be able to support these objectives by providing more accurate, reasonable, and optimal value estimation of a property.

## 3. Data Requirements

The value that we want to predict is the value (price) of a property. The required information to make a prediction are the features of the house 
(e.g., gross building area, the number of rooms, the number of bedrooms, etc), the condition of the house, the location, etc.

## 4. Analytic Approach
### Machine Learning Techniques
Since the value (price) that we want to predict is a continuous value, this problem can be addressed with Supervised Learning, more specifically Regression, 
and the approach to generalization is a model-based learning. We will feed the data in the training set to Regression algorithms which then will tune some 
parameters to fit the model, and then hopefully it will be able to make good predictions on future/unknown instances.

### Risk
There are two possible risk that may be caused by wrong prediction from the ML model:
- The first scenario is when the actual value of the property is higher than the prediction value (the model gives under appraised[10] value), this could cause MPL Bank to reject giving loan and thus resulting in loss of potential borrower.
- The second scenario is when the actual value of the property is lower than prediction value, this could cause MPL Bank to suffer loss when the borrower is unable to pay back the loan.

### Performance Measure
The performance measures to evaluate the ML model are Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Median Absolute Error and the Coefficient of Determination (R²). 

## 5. Action
The business user can utilize the prediction result by comparing it with the appraisal value given by the AMC to determine a reasonable property value.

## 6. Value

The values created from the project are the improvement in the underwriting process and the maximized profit from giving the right appraisal and making 
the right decision in providing loan.

#### Additional Value
It is well known that there are a lot of people who treat property market as an investment instrument and there are cases in which an investor and an appraiser collide to lower the property value in order to help the investor to acquire the property. Thus, we hope this project could also protect <b>property owners</b> by providing fair and reliable property valuation prediction.

---

# II. Data Understanding

Started with importing the dataset which is DC_properties.csv. Then, we continued to handle missing datapoints, give restriction to the data that will be used for EDA and further processes, remove outliers, and choose relevant features to be used in the modelling phase. This process results in a new clean dataset that would be used for this project.

Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/EDA%20-%20Data%20Preparation.ipynb'>EDA - Data Preparation.jpynb</a>

---
  
# III. Data Preparation

At feature engineering, we focused on relevant features by checking and removing outliers further, encoding categorical columns, merging some categorical value which only have a very little sample, checking the correlation in each features, conducting ANOVA F Test, etc.

Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/EDA%20-%20Data%20Preparation.ipynb'>EDA - Data Preparation.jpynb</a>

---
  
# IV. Modeling

In the modelling process the first step we took was splitting the dataset into training, validation, and test set. Then, we tried several base models and ensemble models, with and without hyperparameter tuning. After we obtained the best model for this project, we did residual analysis to evaluate the model performance. 

For the modeling process we tried these base and ensemble models to obtain a model which results in the best scoring metrics: 
  - Linear Regression
  - Polynomial Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Voting Regressor
  - CatBoost Regressor
  
Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/Modeling_Final%20with%20CatBoost%20(without%20Onehot).ipynb'>Modeling_Final with CatBoost (without Onehot).jpynb</a>

---
  
# V. Evaluation

We use CatBoost Regressor as our final model as it gives the best r-squared (R²) results in both training and validation set and the lowest MAE compared with other models.
In this project, we are expected by the business user to make an initial model with Mean Absolute Error (MAE) metrics below ±12% of the mean property price and Median Absolute Error below ±12% of the median property price.


<p float="center ">
<img src="https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Model/Model.png" width="500" />
  
Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Jupyter%20Notebook/Modeling_Final%20with%20CatBoost%20(without%20Onehot).ipynb'>Modeling_Final with CatBoost (without Onehot).jpynb</a>

---
  
# VI. Deployment & Feedback

## 1. Deployment

Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/tree/main/Dashboard'>Dashboard Folder</a>

## 2. Maintenance Plan

We implemented batch (offline) learning in our system since we can't obtain new data autonomously. Thus, if we want the system to know about new data, we have to re-train it with both old and new data and then replace the old system with the new one.


## 3. Final Presentation
Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/tree/main/Presentation'>Presentation Folder</a>


## 4. Project Feedback
Please refer to: <a href='https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/tree/main/Feedback'>Feedback Folder</a>

---
# Webpages
---
HOMEPAGE
---
![](https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Presentation/Webpage%20Screenshots/01-home.png)

ABOUT PAGE
---
![](https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Presentation/Webpage%20Screenshots/02-about.png)

ESTIMATOR PAGE 
---
![](https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Presentation/Webpage%20Screenshots/04-estimator-form-filled.png)

PREDICTION RESULT
---
![](https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Presentation/Webpage%20Screenshots/05-result.png)

INSIGHTS PAGE 
---
![](https://github.com/ls-cy/Purwadhika-JCDS-Final-Project/blob/main/Presentation/Webpage%20Screenshots/06-insights.png)
---

