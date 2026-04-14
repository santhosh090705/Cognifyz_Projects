# Cognifyz Technologies - Data Science Internship 🚀

Welcome to my Data Science Internship project repository! This repository contains the code, dataset, and generated visualizations for the tasks I completed during my internship with **Cognifyz Technologies**. The project involves analyzing a comprehensive restaurant dataset, performing geospatial analysis, discovering customer preferences, and building machine learning models to predict restaurant ratings.

## 🛠️ Tech Stack Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Machine Learning:** Scikit-Learn (Random Forest, Decision Tree, Linear Regression)

## 📁 Project Structure

The project is broken down into multiple levels and tasks, with each script executing a specific part of the data pipeline:

### Level 1: Data Exploration & Preprocessing
* **`Level1_Task1_DataExploration.py`**: Explores the dataset, handles missing values, and performs categorical data conversion. Investigates the class imbalance in aggregate ratings.
* **`Level1_Task2_DescriptiveAnalysis.py`**: Generates high-level statistical measures, uncovers top cities and cuisines, and tracks rating distributions.
* **`Level1_Task3_GeospatialAnalysis.py`**: Maps out the global locations of the restaurants and identifies correlation patterns between geography and customer ratings.

### Level 3: Advanced Analytics & Machine Learning
* **`Level3_Task1_PredictiveModeling.py`**: Implements 3 different Machine Learning Regression models to accurately predict restaurant aggregate ratings. Extracts feature importance (Number of Votes is the strongest indicator).
* **`Level3_Task2_CustomerPreference.py`**: Mines text data to find the most popular cuisines by volume and by highest average customer rating natively.
* **`Level3_Task3_DataVisualization.py`**: Deep dive charting providing comprehensive multi-dimensional data plots covering prices, online delivery, table booking, and restaurant density over geographic zones.

## 📊 Visual Dashboards 

Each Python script inherently generates and saves an insightful visualization dashboard `.png` file showcasing the statistical breakdown. Just run the script, and the dashboard will pop up interactively!

## 🚀 How to Run Locally

1. Clone this repository directly to your machine:
   ```bash
   git clone https://github.com/santhosh090705/Cognifyz_Projects.git
   ```
2. Navigate into the directory and install the following Python packages if not installed:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Run any level/task sequentially via terminal or your code editor to generate predictive insights and data visuals:
   ```bash
   python Level1_Task1_DataExploration.py
   python Level3_Task1_PredictiveModeling.py
   ```

---
*Developed by Santhosh as part of the Cognifyz Technologies Data Science Internship Program.*
