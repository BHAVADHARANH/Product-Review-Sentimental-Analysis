Sentiment Analysis on Customer Reviews
======================================
This project performs Sentiment Analysis on customer reviews to classify them as Positive, Neutral, or Negative. It leverages Natural Language Processing (NLP) techniques and Machine Learning models to achieve accurate predictions.

Project Overview:
The primary goal of this project is to:
-Build a reliable sentiment analysis model for customer reviews.
-Gain insights into customer opinions and product preferences.
-Provide a foundation for applications like customer feedback systems and product recommendation engines.

Project Workflow:
1. Data Loading and Preprocessing:

Loads data from "1429_1.csv" using Pandas.
Handles missing values.
Splits data into training and testing sets using Stratified Sampling to ensure balanced sentiment classes.

2. Exploratory Data Analysis (EDA):

Analyzes product IDs (ASINs) and their relationship with customer ratings.
Visualizes data patterns using Matplotlib and Seaborn.
Identifies data distribution and potential biases.

3. Feature Engineering:

Converts numerical ratings into sentiment categories:
Positive: Ratings 4 and 5
Neutral: Rating 3
Negative: Ratings 1 and 2
Extracts text features using:
CountVectorizer
TF-IDF Transformer

4. Model Training and Evaluation:
Implemented Machine Learning models:
Multinomial Naive Bayes
Logistic Regression
Linear Support Vector Classifier (LinearSVC)
Decision Tree Classifier
Random Forest Classifier
Evaluation Metrics:
Accuracy Score
Classification Reports

5. Hyperparameter Tuning
Utilizes GridSearchCV to find the best hyperparameters for the LinearSVC model, improving model performance.

6. Predictions
Demonstrates sentiment prediction on new customer review inputs.

Library  -      pupose
---------------------------------------------
Pandas         Data manipulation and analysis
Matplotlib     Data visualization
Seaborn        Statistical data visualization
NumPy          Numerical computations
Scikit-learn   Machine learning models and pipelines

How to Run the Project:

Install required libraries:
pip install pandas matplotlib seaborn scikit-learn
Run the main notebook or Python file:
python sentiment_analysis.py
View model performance and predictions.
Results:
The best model achieved 92% accuracy using LinearSVC.
Sentiment analysis insights can help businesses improve customer satisfaction.

License
This project is for educational purposes only.



