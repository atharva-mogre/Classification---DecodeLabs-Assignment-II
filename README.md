# Iris Flower Classification using Machine Learning

This project implements a foundational machine learning workflow to solve the classic Iris flower classification problem. Using Python and the Scikit-learn library, the script is designed to categorize Iris flowers into their correct species based on their sepal and petal dimensions. This repository serves as a practical demonstration of supervised learning, making it a great resource for understanding data splitting, model training, and performance evaluation.

The script explores two distinct classification algorithms to provide a comparative analysis of their predictive capabilities. First, it utilizes a Random Forest Classifier, an ensemble learning method known for its robustness and accuracy. Second, it implements a Logistic Regression model to offer a simpler, linear approach to the classification task. By evaluating both models on a dedicated testing dataset, the project provides a comprehensive look at how different algorithms tackle the exact same data.

## Key Highlights
- **Standardized Dataset**: Leverages the classic Iris dataset, loaded directly via Scikit-learn, ensuring reliable and reproducible results.
- **Model Comparison**: Trains and evaluates both Random Forest and Logistic Regression algorithms side-by-side.
- **Comprehensive Evaluation**: Outputs not just basic accuracy scores, but also detailed classification reports including precision, recall, and f1-scores for deep analysis.
- **Standard ML Pipeline**: Demonstrates best practices by splitting the data into an 80% training set and a 20% testing set.

## Getting Started
Ensure you have Python installed along with the required libraries (`pandas` and `scikit-learn`). You can run the classification script via your terminal:
`python Classification.py`
