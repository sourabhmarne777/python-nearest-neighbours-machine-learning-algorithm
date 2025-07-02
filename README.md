# Python Nearest Neighbours Machine Learning Algorithm

A custom implementation of the K-Nearest Neighbors (KNN) algorithm from scratch using Python and NumPy, with comparison to scikit-learn's built-in implementation.

## Overview

This project demonstrates a complete implementation of the 1-Nearest Neighbor algorithm without using any machine learning libraries for the core logic. The implementation includes custom distance calculation, prediction logic, and performance evaluation on Iris and Ionosphere datasets.

## Results

### Iris Dataset
- **Custom Implementation Accuracy**: 97.37%
- **Scikit-learn Accuracy**: 97.37%
- **Test Error Rate**: 2.63%

### Ionosphere Dataset
- **Custom Implementation Accuracy**: 87.50%
- **Scikit-learn Accuracy**: 87.50%
- **Test Error Rate**: 12.50%

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sourabhmarne777/python-nearest-neighbours-machine-learning-algorithm.git
cd python-nearest-neighbours-machine-learning-algorithm
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook "Nearest Neighbour Algorithm.ipynb"
```

## Usage

Open the Jupyter notebook and run all cells to see the complete implementation. The notebook will load datasets, run the custom KNN implementation, compare results with scikit-learn, and display accuracy metrics.

## Core Algorithm

The `NNpredict` function implements 1-Nearest Neighbor using Euclidean distance calculation to find the closest training sample for each test sample.

## Dataset Information

- **Iris Dataset**: 4 features, 3 classes, 150 samples
- **Ionosphere Dataset**: 5 features (from 34), 2 classes, binary classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Sourabh Marne**
- GitHub: [@sourabhmarne777](https://github.com/sourabhmarne777)