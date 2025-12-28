# Data-Pipeline-Project
# Adult Income Analysis - KNN & Plotly

A Python GUI project for analyzing the Adult Income dataset using **K-Nearest Neighbors (KNN)** classification and interactive visualizations with **Plotly**. The GUI is built with **Tkinter**, allowing data cleaning, outlier removal, plotting, KNN modeling, and predicting new data.

---

## Features

- **Load CSV**: Load the dataset and display basic info (`head`, `tail`, `describe`, `info`).  
- **Clean Data**: Handle missing values (`?`) and drop rows with critical missing columns.  
- **Remove Outliers**: Remove outliers from numeric columns (`age`, `capital.gain`, `capital.loss`, `hours.per.week`).  
- **Save Cleaned Data**: Save the cleaned dataset as a CSV file.  
- **Static Plots**: Boxplots and histograms for numeric columns (before/after cleaning).  
- **Interactive Plots**: Plotly-based boxplots and histograms for numeric columns.  
- **KNN Classification**: Train a K-Nearest Neighbors model with a user-defined `k`. Shows accuracy and classification report.  
- **Predict New Data**: Input new data in a GUI window and predict income using the trained KNN model.

---

## Requirements

- Python 3.8+
- Required Python packages:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn plotly
