# Cracking the Code of Bank’s Marketing Campaign: A Predictive Modelling Approach

Welcome to the GitHub repository for the project **"Cracking the Code of Bank’s Marketing Campaign: A Predictive Modelling Approach"**. This repository contains the code and resources used in the analysis of a Portuguese bank's direct phone call marketing campaigns. The project aimed to predict and enhance the success of term deposit promotions among existing customers. 

## Project Overview

In this project, we undertook a comprehensive analysis of the bank's marketing campaigns carried out from May 2008 to November 2010. The project encompassed data preprocessing, outlier handling, exploratory data analysis, data visualization, supervised and unsupervised machine learning, and insights generation.

### Key Steps

1. **Data Cleaning and Preprocessing:** The raw data was meticulously cleaned to ensure accurate and reliable analysis. We addressed missing values and inconsistencies to create a reliable dataset.

2. **Exploratory Data Analysis (EDA):** Before diving into machine learning, we conducted thorough EDA. We created histograms, bar charts, scatter plots, and box plots to gain insights into variable distributions, identify outliers, and explore potential relationships between variables.

3. **Supervised Learning Models:** We implemented a range of supervised learning algorithms to predict campaign success. The models included:
   - Logistic Regression
   - Decision Tree Classifier
   - Naive Bayes Classifier
   - Random Forest Classifier
   - Gradient Boosting Classifier
   - Support Vector Machine
   - Vote Classifier

4. **Identifying the Best Model:** Upon thorough evaluation of the models, the Random Forest Classifier emerged as the leading performer. This result was achieved through meticulous application of SMOTE sampling and data normalization during the feature engineering process. The classifier demonstrated an outstanding accuracy rate of 90% and secured the highest F1 score of 57, surpassing the performance of other algorithms in the comparison.

5. **Unsupervised Learning Methods:** We delved into unsupervised learning techniques to uncover hidden patterns:
   - Principal Component Analysis (PCA)
   - K-Means Clustering
   - K Nearest Neighbors
   - Hierarchical Clustering
   - Anomaly Detection
   - Neural Networking

6. **Visualizing Clusters and Inferences:** Leveraging the insights from unsupervised learning, we created informative visualizations. These visualizations were showcased alongside the results of the analysis on a dynamic dashboard developed using Streamlit. Our visualizations and inferences provide actionable insights for decision-makers. We highlighted clusters with high subscription rates, suggesting targeting these clusters to boost sales.

## Repository Structure

- `data/`: This directory contains the dataset used in the project.
- `notebooks/`: Jupyter notebooks detailing each step of the analysis, from data cleaning to machine learning and visualization.
- `dashboard/`: Source code and scripts for the Streamlit dashboard.

## Dashboard

The core findings and insights of this project are brought to life through an interactive dashboard developed using Streamlit and Plotly. The dashboard enables users to visualize the results of the analysis in a dynamic and intuitive manner. By seamlessly integrating visualizations and clustering results, the dashboard provides a holistic view of the marketing campaign's performance and offers actionable recommendations.

## Citation:
  This dataset is public available for research.

  [Moro et al., 2011] S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology. 
  In P. Novais et al. (Eds.), Proceedings of the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal, October, 2011. EUROSIS.

  Available at: - [pdf] http://hdl.handle.net/1822/14838
                - [bib] http://www3.dsi.uminho.pt/pcortez/bib/2011-esm-1.txt

