import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neural_network import MLPClassifier

from spike_functions import Spike_object


class SpikeSortingMLP:
    """
    A class for sorting spikes using a Multi-Layer Perceptron (MLP) classifier.
    """
    def __init__(self):
        """
        Initializes the SpikeSortingMLP class with a configured MLPClassifier.
        """
        self.n = MLPClassifier(
            hidden_layer_sizes=(20,),
            random_state=1,
            max_iter=1000,
            activation='relu',
            solver='adam',
            learning_rate_init=0.01,
            batch_size=16,
            verbose=False
        )      

    def load_and_prepare_data(self, file_path, train=False):
        """
        Loads and prepares spike data from a given file.

        Parameters:
        - file_path (str): The path to the .mat file containing spike data.
        - train (bool): Whether to load training data (includes index and class data).

        Returns:
        - Spike_object: The loaded and prepared spike data.
        """
        spike_data = Spike_object()
        try:
            spike_data.load_data(file_path, train=train)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None

        if train:
            spike_data.sort()
            spike_data.filter(2500, 'low')
        
        return spike_data  

    def train_mlp(self):
        """
        Trains the MLP classifier using spike data loaded from a .mat file.
        The method also prepares a validation data set from the training data.
        """
        # Load and prepare training data
        training_file = 'D1.mat'
        self.training_data = self.load_and_prepare_data(training_file, train=True)
        if self.training_data is None:
            return

        # Split data for validation
        self.validation_data = self.training_data.split_data(0.2)

        # Run spike detection and comparison on training data
        self.training_data.compare()

        # Train the MLP with training dataset classes
        self.n.fit(self.training_data.create_window(), self.training_data.classes)


    def validate_mlp(self):
        """
        Validates the trained MLP classifier using the validation data set.
        Outputs the spike detection score, class detection score, and overall score.
        """
        if self.validation_data is None:
            print("Validation data not available.")
            return

        # Run spike detection and comparison on validation data
        spike_score = self.validation_data.compare()

        # Classify detected spikes and compare to known classes
        predicted = self.n.predict(self.validation_data.create_window())
        classified = np.where(predicted == self.validation_data.classes)[0]

        # Score classifier method and display performance metrics
        class_score = (len(classified) / len(self.validation_data.index))
        print(f'Spike detection score: {spike_score:.4f}')
        print(f'Class detection score: {class_score:.4f}')
        print(f'Overall score: {(spike_score * class_score):.4f}')

        cm = confusion_matrix(self.validation_data.classes, predicted)
        print(cm)
        cr = classification_report(self.validation_data.classes, predicted, digits=4)
        print(cr)

        return class_score

    def submission(self):
        """
        Prepares and exports the submission data set based on the trained MLP model.
        Predicts spike classes and exports them in a .mat file.
        """
        submission_file = 'D1.mat'
        self.submission_data = self.load_and_prepare_data(submission_file)
        if self.submission_data is None:
            return

        # Additional processing specific to submission data
        self.submission_data.filter([25, 1900], 'band')
        self.submission_data.find_spikes()

        # Predict classes using the trained MLP model
        predicted = self.n.predict(self.submission_data.create_window())

        # Display class breakdown
        print('Class Breakdown')
        breakdown = self.class_breakdown(predicted)

        # Export predictions to a .mat file
        mat_file = {'Index': self.submission_data.index, 'Class': predicted}
        try:
            savemat('D1_mlp.mat', mat_file)
            print("Submission data saved successfully.")
        except Exception as e:
            print(f"Error saving submission data: {e}")

    def class_breakdown(self, classes):
        """
        Calculates and prints the breakdown of spike classes.

        Parameters:
        - classes (np.array): Array of classified spike types.

        Returns:
        - dict: A dictionary with class types as keys and their counts as values.
        """
        unique, counts = np.unique(classes, return_counts=True)
        breakdown = dict(zip(unique, counts))

        for key, val in breakdown.items():
            print(f'Type {key:g}: {val}')
        return breakdown


if __name__ == '__main__':

    s = SpikeSortingMLP()
    s.train_mlp()
    s.validate_mlp()
    s.submission()


    


