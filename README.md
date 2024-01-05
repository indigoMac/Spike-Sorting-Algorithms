# Spike-Sorting-Algorithms
This repository contains Python scripts for spike sorting using two distinct machine learning algorithms: K-Nearest Neighbors (KNN) and Multi-Layer Perceptron (MLP). These methods are applied to classify neural spikes, a critical task in the field of neuroscience

## Files in the Repository
- **KNN_main.py**: Implements spike sorting using the K-Nearest Neighbors algorithm. This script manages data loading, preprocessing, and classification using KNN.
- **mlp_main.py**: Utilizes a Multi-Layer Perceptron, an artificial neural network, for spike classification. It handles data loading, preprocessing, and the application of the MLP model.
- **spike_functions.py**: Provides various utility functions for processing spike data. These functions are employed by both KNN_main.py and mlp_main.py.

## Datasets
Included datasets in .mat format:

- **D1.mat**: Contains spike data with known validation data, primarily used for training and testing the algorithms.
- **D2.mat**, **D3.mat**, **D4.mat**: Contain spike data without validation data, suitable for further experimentation.

## Algorithm Comparison
- K-Nearest Neighbors (KNN): This method is based on feature similarity, where the classification of a new spike is determined by the majority class of its nearest neighbors. It's generally simpler and can be very effective for smaller datasets.

- Multi-Layer Perceptron (MLP): As a neural network approach, MLP can model more complex patterns and relationships in the data. It is more suited for larger datasets and can achieve higher accuracy, albeit at the cost of increased computational complexity and the need for more data for training.

## Getting Started
1. Clone the repository to your local machine.
2. Install Python and necessary libraries: numpy, scipy, matplotlib, pandas, scikit-learn.
3. Place the .mat data files in the same directory as the scripts.
4. Execute KNN_main.py or mlp_main.py as per your requirement.

## Usage
- To use the KNN approach, run the KNN_main.py script.
- For the MLP model, execute the mlp_main.py script.

Each script will load the data, perform preprocessing, and classify the spikes according to the chosen algorithm.

## Contributing
Contributions, suggestions, and improvements are welcome. Feel free to fork this repository and submit your pull requests.
