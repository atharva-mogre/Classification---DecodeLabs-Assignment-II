# Iris Flower Classification using Machine Learning

This project implements a foundational machine learning workflow to solve the classic Iris flower classification problem. Using Python and the Scikit-learn library, the script is designed to categorize Iris flowers into their correct species based on their sepal and petal dimensions. This repository serves as a practical demonstration of supervised learning, making it a great resource for understanding data standardization, model training, and diagnostic performance evaluation.

The script utilizes the K-Nearest Neighbors (KNN) algorithm to provide a robust predictive framework for the classification task. By applying feature scaling via StandardScaler, it normalizes the input dimensions, ensuring optimal algorithmic efficiency and accuracy. Evaluating the model on a dedicated testing dataset, the project provides a comprehensive look at how pattern recognition works in supervised learning environments.

## Key Highlights
- **Standardized Dataset**: Leverages the classic Iris dataset, loaded directly via Scikit-learn, ensuring reliable and reproducible results.
- **Feature Scaling**: Applies StandardScaler to balance feature weights and eliminate magnitude bias.
- **KNN Architecture**: Employs the K-Nearest Neighbors (KNN) algorithm with K=5 to classify unseen data points based on proximity.
- **Comprehensive Evaluation**: Outputs the Accuracy Score, a detailed Classification Report (precision, recall, f1-scores), and a Confusion Matrix for deeper diagnostic insights.
- **Standard ML Pipeline**: Demonstrates best practices by splitting the data into an 80% training set and a 20% testing set.

## Getting Started
Ensure you have Python installed along with the required libraries (`pandas` and `scikit-learn`). You can run the classification script via your terminal:
`python Classification.py`
