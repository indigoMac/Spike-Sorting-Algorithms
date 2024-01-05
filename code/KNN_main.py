import os
import numpy as np
from scipy.io import savemat, loadmat
from scipy.optimize import dual_annealing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
# import class with all made functions
from spike_functions import Spike_object

class Spike_sort_KNN:
    """
    A class to handle the training and validation of a K-Nearest Neighbors classifier for spike sorting.
    """
    def __init__(self, verbose=True):        
        """
        Initializes the SpikeSortKNN class.

        Parameters:
        verbose (bool): If True, enables verbose output.
        """
        self.n = KNeighborsClassifier(n_neighbors=5, p=2)
        self.verbose = verbose
        self.target = 1
        self.training = None
        self.validation = None
        #self.folder_name = 'Results5'
        #os.mkdir(self.folder_name)
        
    def train(self, training_file='D1.mat'):       
        """
        Trains the KNN classifier using the specified training data file.

        Parameters:
        training_file (str): Path to the .mat file containing training data.
        """
        # initiate Spike_object for training data
        self.training = Spike_object()

        # retrieve data in the training data set  
        self.training.load_data(training_file, train=True)

        # Sort index/class data 
        self.training.sort()
        
        # Take last 20% of training data set and use as a validation data set
        self.validation = self.training.split_data(0.2)

        # Filter the input data
        self.training.savitzky_golay_filter(25, 5)
        self.validation.savitzky_golay_filter(25, 5)
        
        # Run spike detection and comparison on training data
        self.training.compare()

        # train nn with training dataset known classes        
        self.n.fit(self.training.create_window(), self.training.classes)

    # measure performace of nn
    def validate(self):     
        """
        Validates the trained KNN classifier using the validation data.

        Returns:
        float: The classification score.
        """
        if self.validation is None:
            print("Validation data is not available.")
            return 0
        
        # spike detection and comparison on validation data
        spike_score = self.validation.compare()  

        #mean_score, std_dev = self.validation.cross_validate(self.n)

        # predict class detected spikes
        predict = self.n.predict(self.validation.create_window())

        # compare to known classes
        classified = np.where(predict == self.validation.classes)[0]

        # score output
        class_score = (len(classified) / len(self.validation.index))

        if self.verbose:
            self._print_validation_results(spike_score, class_score, predict)

        return class_score
    
    def _print_validation_results(self, spike_score, class_score, predict):
        """
        Prints the validation results.

        Parameters:
        spike_score (float): The spike detection score.
        class_score (float): The classification score.
        predict (np.array): Predicted classes.
        """
        print(f'Spike detection score: {spike_score:.4f}')
        print(f'Class detection score: {class_score:.4f}')
        print(f'Overall score: {(spike_score * class_score):.4f}')
        confusion = confusion_matrix(self.validation.classes, predict)
        print(confusion)
        class_results = classification_report(self.validation.classes, predict, digits=4)
        print(class_results)

    def output(self, input_file, output_file, file_num=3):
        """
        Processes the input file using the trained classifier and saves the output.

        Parameters:
        input_file (str): Path to the input .mat file.
        output_file (str): Path for saving the output .mat file.
        file_num (int): Identifier for the type of processing required for the file.
        """
        self.outputs = Spike_object()
        self.outputs.load_data(input_file, train=False)

        if file_num == 1:
            optimal_frequencies = self._find_optimal_frequencies()
            self.outputs.filter([optimal_frequencies.x[0], optimal_frequencies.x[1]], 'band')
        elif file_num == 2:
            self.outputs.savitzky_golay_filter(25, 5)
        elif file_num == 3:
            self.outputs.filter([10,2000], 'band')

        spikes = self._find_spikes(file_num)
        print(f'{len(spikes)} spikes detected')
        predict = self.n.predict(self.outputs.create_window())
        self.outputs.classes = predict

        self.class_breakdown(predict)
        self._save_output(output_file)

    def _find_optimal_frequencies(self):
        """
        Finds the optimal frequencies for filtering the signal.

        Returns:
        Result of the optimization algorithm.
        """
        da_iterations = 1000
        bounds = [[50,150], [2000,3500], [1,6], [25,150], [25000,50000]]
        optimal_frequencies = dual_annealing(self.optimise_filtering, bounds, maxfun=da_iterations)
        print('Best Result  = ', optimal_frequencies.fun)
        return optimal_frequencies

    def _find_spikes(self, file_num):
        """
        Identifies spikes in the data based on the specified file number.

        Parameters:
        file_num (int): Identifier for the type of spike detection to use.

        Returns:
        np.array: Indices of detected spikes.
        """
        if file_num == 1:
            return self.outputs.detect_peaks(optimal_frequencies.x[2], optimal_frequencies.x[3])
        elif file_num == 2:
            return self.outputs.find_spike()
        else:
            return self.outputs.find_spikes()

    def _save_output(self, output_file):
        """
        Saves the processed data to a .mat file.

        Parameters:
        output_file (str): Path for saving the output .mat file.
        """
        mat_file = {'D': self.outputs.data, 'Index': self.outputs.index, 'Class': self.outputs.classes}
        savemat(output_file, mat_file)


    def class_breakdown(self, classes):
        """
        Prints a breakdown of the number of spikes per class.

        Parameters:
        classes (np.array): Array of spike classes.

        Returns:
        dict: Breakdown of spike classes and their counts.
        """
        unique, counts = np.unique(classes, return_counts=True)
        breakdown = dict(zip(unique, counts))
        for key, val in breakdown.items():
            print(f'Type {key:g}: {val}')
        return breakdown

    
    def optimise_filtering(self, Inputs):
        """
        Optimizes filtering parameters for spike detection.

        Parameters:
        Inputs (list): A list of parameter values to be optimized.

        Returns:
        float: Score representing the optimization objective.
        """
        low_cut, high_cut, xprominence, wlen, sampling_freq = Inputs
        spikes = self.outputs.find_spikes(low_cut, high_cut, xprominence, wlen, sampling_freq)
        return abs(self.target - len(spikes))


if __name__ == '__main__':
    ####### task 1 and 2 ######
    s = Spike_sort_KNN(verbose=True)
    s.train()
    s.validate()
    s.output('D2.mat', 'D2.mat', 2)

    ####### task 3 #########
    s.output('D3.mat', 'D3.mat', 3)

    ####### task 4 #########
    s.output('D4.mat', 'D4.mat', 3)