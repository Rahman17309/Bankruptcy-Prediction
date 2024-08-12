# Bankruptcy Prevention Prediction Project

# Video:
[![Bankruptcy Prevention Prediction](https://ytcards.demolab.com/?id=jHw8JbTdTC4&title=Bankruptcy+Prevention+Prediction&lang=en&timestamp=1723401000&background_color=230d1117&title_color=23ffffff&stats_color=%23dedede&max_title_lines=16width=250&border_radius=5&duration=311 "Bankruptcy Prevention Prediction")](https://youtu.be/jHw8JbTdTC4)


# Overview

This project involves building and evaluating machine learning models to predict the likelihood of bankruptcy in companies. Various algorithms were explored, including Logistic Regression, Random Forest, Gradient Boosting, Support Vector Machines (SVM), and Decision Trees. The project also includes hyperparameter tuning using GridSearchCV and RandomizedSearchCV to optimize model performance.

# Project Structure

data/: Contains the dataset used for training and testing the models.

notebooks/: Jupyter notebooks with detailed Exploratory Data Analysis (EDA), model building, and evaluation.

models/: Serialized models saved as .pkl files, including both GridSearchCV and RandomizedSearchCV versions for Logistic Regression, Random Forest, and SVM.

app/: Streamlit application code for deploying the models.

README.md: This file, providing an overview of the project.

requirements.txt: List of Python dependencies required to run the project.

# Dataset

The dataset used for this project is a financial dataset with features that capture various risk factors such as industrial risk, management risk, financial flexibility, credibility, competitiveness, and operating risk. The target variable is a binary indicator of whether a company went bankrupt.

# Features:

1. industrial_risk: Risk associated with the industry sector.

2. management_risk: Risk related to management decisions.

3. financial_flexibility: Ability to adapt financially to adverse conditions.

4. credibility: Company's trustworthiness in the market.

5. competitiveness: Company's ability to compete in its sector.

6. operating_risk: Risk associated with day-to-day operations.

# Exploratory Data Analysis (EDA)

EDA was conducted to understand the distribution of the features, detect any potential outliers, and identify correlations between features. Visualizations such as histograms, box plots, and correlation heatmaps were used to gain insights into the data.

# Model Building

In this project, multiple machine learning models were implemented to predict bankruptcy, each with a distinct approach to handling binary classification. Below is a detailed explanation of each model, including the theoretical background, implementation, and hyperparameter tuning strategies.

## Logistic Regression

### Theory

Logistic Regression is a linear model used for binary classification tasks. It estimates the probability of a binary outcome based on one or more predictor variables by applying the logistic function. This function transforms the linear combination of inputs into a probability score, which can then be used to predict class labels.

### Implementation

The Logistic Regression model was implemented using the LogisticRegression class from the scikit-learn library. The model was trained on the full dataset, and the coefficients of the features were analyzed to understand their impact on bankruptcy predictions.

### Hyperparameter Tuning

GridSearchCV: An exhaustive search was conducted over a specified parameter grid to find the best hyperparameters for the Logistic Regression model. The parameters tuned included:

1. C: The inverse of regularization strength, controlling the trade-off between achieving a low training error and a low testing error.
2. solver: The algorithm used to optimize the model parameters.

RandomizedSearchCV: A randomized search over specified parameter distributions was also conducted. This approach is less computationally intensive than GridSearchCV and often yields similarly optimal results. The same parameters as in GridSearchCV were tuned but with random sampling.

### Rationale

Logistic Regression is favored for its simplicity, interpretability, and efficiency, particularly when the relationship between features and the target variable is linear. It is a robust baseline model for binary classification tasks like bankruptcy prediction.

## Random Forest

### Theory

Random Forest is an ensemble learning technique that builds multiple decision trees during training and outputs the mode of the classes (classification) of the individual trees. By averaging the results of many decision trees, Random Forest improves prediction accuracy and robustness, reducing the risk of overfitting.

### Implementation

The RandomForestClassifier from scikit-learn was used to build the Random Forest model. Each tree in the forest was trained on a bootstrapped subset of the data, and predictions were aggregated to make the final classification. Feature importance metrics were also extracted from the model, providing insights into which features most influenced the predictions.

### Hyperparameter Tuning

GridSearchCV: The following hyperparameters were fine-tuned using GridSearchCV:

1. n_estimators: Number of trees in the forest.
2. max_depth: Maximum depth of the trees, controlling the complexity of the model.
3. min_samples_split: Minimum number of samples required to split an internal node.
4. RandomizedSearchCV: To explore a broader range of hyperparameters more efficiently, RandomizedSearchCV was employed. This method randomly sampled from a specified set of distributions for the above parameters.

### Rationale
Random Forest is particularly well-suited for this problem due to its ability to handle a large number of input variables and its robustness against overfitting. The model's built-in feature importance scores also provide valuable insights into the data.

## Gradient Boosting

### Theory

Gradient Boosting is an ensemble technique that builds models sequentially, with each new model attempting to correct the errors made by the previous ones. By focusing on the instances that are harder to predict, Gradient Boosting can create a strong predictive model from a collection of weak learners (typically decision trees).

### Implementation

The GradientBoostingClassifier from scikit-learn was used to implement this model. The model was trained by adding trees sequentially, with each new tree improving the predictions of the ensemble.

### Hyperparameter Tuning

GridSearchCV: The key hyperparameters tuned included:

1. n_estimators: Number of boosting stages.
2. learning_rate: How much each tree contributes to the final model.
3. max_depth: Maximum depth of the individual trees.
4. RandomizedSearchCV: Used to randomly sample a range of hyperparameter values to find the best configuration.

### Rationale

Gradient Boosting is highly effective for datasets where the relationship between features and the target variable is complex and non-linear. Its sequential nature allows it to focus on the most challenging instances, leading to a highly accurate model.

## Support Vector Machines (SVM)

### Theory

Support Vector Machines (SVM) are powerful classifiers that work by finding the hyperplane that best separates the classes in the feature space. The SVM algorithm aims to maximize the margin between the closest points of the classes, known as support vectors, and the hyperplane.

### Implementation

The SVM model was implemented using the SVC class from scikit-learn. Both linear and non-linear kernels were explored to capture the relationship between features and bankruptcy likelihood.

### Hyperparameter Tuning

GridSearchCV: Tuning focused on the following parameters:

1. C: Regularization parameter controlling the trade-off between achieving a large margin and classifying training points correctly.
2. kernel: Specifies the kernel type to be used in the algorithm (e.g., linear, polynomial, RBF).
3. gamma: Kernel coefficient for non-linear hyperplanes.
4. RandomizedSearchCV: This method was also used to explore a wide range of hyperparameters, particularly for non-linear kernels.

### Rationale

SVM is particularly effective in high-dimensional spaces and is versatile due to its ability to use different kernel functions. It's a strong choice for classification problems where the decision boundary is not easily separable by a linear model.

## Decision Trees

### Theory

Decision Trees are a non-parametric supervised learning method used for classification and regression tasks. The model splits the data into subsets based on the value of input features, creating a tree-like structure where each node represents a decision based on a feature.

### Implementation

The DecisionTreeClassifier from scikit-learn was employed to build the Decision Tree model. The model was trained on the entire dataset, and various metrics like Gini impurity and entropy were used to determine the best splits.

### Hyperparameter Tuning

1. GridSearchCV: The following parameters were optimized:
2. max_depth: Maximum depth of the tree.
3. min_samples_split: Minimum number of samples required to split a node.
4. criterion: The function to measure the quality of a split (Gini or entropy).

### Rationale
Decision Trees are simple to understand and interpret, making them a good baseline model for comparison with more complex models. Although prone to overfitting, they are useful for gaining quick insights into the data and can be improved with techniques like pruning.



# Model Evaluation

The models were evaluated using various metrics:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
Accuracy: Proportion of correctly predicted instances.
Precision, Recall, and F1-Score: To evaluate the model's performance, particularly in handling imbalanced data.
Hyperparameter Tuning
GridSearchCV and RandomizedSearchCV were used to optimize the hyperparameters for the models. The tuned models were then compared based on their performance metrics.

# Model Deployment

A Streamlit app was created to deploy the models, allowing users to input risk factors and receive a prediction on whether the company is likely to go bankrupt. The app features:

A dropdown menu to select between different models (Logistic Regression, Random Forest, and SVM).
Sliders for input features such as industrial risk, management risk, and others.
A prediction button that displays 'non-bankruptcy' for 0 and 'bankruptcy' for 1.

# Results and Insights
The Random Forest model with GridSearchCV tuning achieved the highest accuracy and F1-Score, making it the most reliable model for predicting bankruptcy in this dataset. The Streamlit app provides an easy-to-use interface for making predictions based on user inputs.

# Conclusion
This project demonstrates the application of various machine learning techniques to predict bankruptcy. The models were rigorously evaluated, and the best-performing model was deployed in an interactive web application. This work can serve as a foundation for further exploration in financial risk prediction.
