# Python Nearest Neighbours Machine Learning Algorithm

A custom implementation of the K-Nearest Neighbors (KNN) algorithm from scratch using Python and NumPy, with comparison to scikit-learn's built-in implementation.

## Overview

This project demonstrates a complete implementation of the 1-Nearest Neighbor algorithm without using any machine learning libraries for the core logic. The implementation includes:

- Custom distance calculation using Euclidean distance
- Prediction logic for classification tasks
- Performance evaluation and comparison with scikit-learn
- Testing on two different datasets: Iris and Ionosphere

## Features

- **Custom KNN Implementation**: Built from scratch using only NumPy for mathematical operations
- **Dataset Support**: Works with both Iris and Ionosphere datasets
- **Performance Metrics**: Calculates accuracy, error count, and error rate
- **Comparison Analysis**: Side-by-side comparison with scikit-learn's KNeighborsClassifier
- **Jupyter Notebook**: Interactive implementation with detailed explanations

## Results

### Iris Dataset
- **Custom Implementation Accuracy**: 97.37%
- **Scikit-learn Accuracy**: 97.37%
- **Test Error Rate**: 2.63%
- **Number of Errors**: 1

### Ionosphere Dataset
- **Custom Implementation Accuracy**: 87.50%
- **Scikit-learn Accuracy**: 87.50%
- **Test Error Rate**: 12.50%
- **Number of Errors**: 11

## Core Algorithm

The heart of this implementation is the `NNpredict` function:

```python
def NNpredict(TrainSample, TrainLabel, test):
    """
    Predicts labels for test samples using 1-Nearest Neighbor algorithm
    
    Parameters:
    - TrainSample: Training feature data
    - TrainLabel: Training labels
    - test: Test feature data
    
    Returns:
    - predicted_label: List of predicted labels for test samples
    """
    # Implementation uses Euclidean distance calculation
    # Finds the closest training sample for each test sample
    # Returns the label of the nearest neighbor
```

## Requirements

- Python 3.x
- NumPy
- scikit-learn
- Jupyter Notebook

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sourabhmarne777/python-nearest-neighbours-machine-learning-algorithm.git
cd python-nearest-neighbours-machine-learning-algorithm
```

2. Install required packages:
```bash
pip install numpy scikit-learn jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook "Nearest Neighbour Algorithm.ipynb"
```

## Usage

1. Open the Jupyter notebook
2. Run all cells to see the complete implementation
3. The notebook will:
   - Load the Iris and Ionosphere datasets
   - Split data into training and testing sets
   - Run the custom KNN implementation
   - Compare results with scikit-learn's implementation
   - Display accuracy metrics and error analysis

## Dataset Information

### Iris Dataset
- **Source**: scikit-learn's built-in dataset
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Samples**: 150

### Ionosphere Dataset
- **Source**: ionosphere.txt file
- **Features**: 5 (selected from 34 available features)
- **Classes**: 2 (binary classification)
- **Purpose**: Radar signal classification

## Algorithm Details

The implementation follows these key steps:

1. **Distance Calculation**: Uses Euclidean distance formula
2. **Neighbor Selection**: Finds the single nearest neighbor (k=1)
3. **Prediction**: Assigns the label of the nearest training sample
4. **Evaluation**: Compares predictions with actual test labels

## Performance Analysis

The custom implementation achieves identical results to scikit-learn's KNeighborsClassifier, demonstrating the correctness of the algorithm. This validates that the mathematical implementation and logic are sound.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
