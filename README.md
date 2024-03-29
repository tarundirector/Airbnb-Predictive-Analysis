
# 🏠 Airbnb Rental Rate Prediction 📈

This project aims to forecast the pricing of Airbnb listings based on various features. It encompasses phases such as delving into the data for exploratory analysis, preprocessing the data, selecting pertinent features, fitting models, comparing them, and ultimately deploying a containerized web application on Heroku. This deployment is facilitated through a continuous integration and continuous deployment (CI/CD) pipeline, employing Docker and GitHub Actions.


## Project Objectives

🏠 The primary goal of this project is to aid Airbnb hosts in determining suitable prices for their listings.

💡 **Problem:** New Airbnb hosts face challenges in determining the optimal price for their listings. Without a convenient tool, hosts often resort to estimating prices based on neighboring listings, leading to potential underpricing or overpricing.

🔍 **Solution:** Introducing a Predictive Price Modeling tool designed to empower new hosts in setting competitive prices for their listings. By leveraging Machine Learning, hosts can input various listing details such as location, property attributes, and available amenities. The model, trained on extensive Airbnb data, then generates a suggested price based on similar listings.

This approach streamlines the pricing process, providing hosts with data-driven insights to maximize their listing's earning potential while ensuring competitiveness in the market.
## Project Implementation Overview

Here's the breakdown of the project steps 🛠️:

1. **Exploratory Data Analysis (EDA):** Delve into the dataset, examining various features' distributions using Histograms and Box-plots 📊.

2. **Pre-processing and Data Cleaning:** Prepare the data for analysis by normalizing, filling missing values, and encoding categorical values 🧹.

3. **Feature Selection:** Identify the most influential features for predicting listing prices by analyzing their correlation with the response variable (Listing Price) 🎯.

4. **Model Fitting and Selection:** Train multiple models, fine-tune hyperparameters, and evaluate model performance using Learning Curves 🤖.

5. **Model Serving:** Utilize Flask to deploy and serve model predictions via a REST API 🚀.

6. **Containerization:** Employ Docker to containerize the web application, ensuring consistent deployment across environments 🐳.

7. **Production Deployment:** Implement a CI/CD pipeline on GitHub for continuous integration and deployment, ensuring smooth updates and maintenance 🛠️. Heroku is utilized as the deployment platform to host the application and make it accessible to users online 🌐.

![Colorful Modern Business Timeline Infographic Graph (1)](https://github.com/tarundirector/airbnb-predictive-analysis/assets/85684655/eface5b5-e921-4b95-b486-8a3e860ea3e9)

## Results

Below is a screenshot of the application in action, capturing the entire interface. Users have the option to input all pertinent details of their listings. Once entered, the trained Predictive Model leverages this information to generate a price estimate for the listing based on its features.

![ezgif-4-67daf127c7](https://github.com/tarundirector/airbnb-predictive-analysis/assets/85684655/e9c966b4-590f-43d1-b58f-a805e841b8d6)
