from scipy.signal import find_peaks, butter, sosfiltfilt, savgol_filter
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.fft import fft, ifft
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#Spike_funcations - store and maniuplate spike data, as well as spike index and class data
class Spike_object:

    # set variables
    def __init__(self, data=None, index=None, classes=None, verbose=True):
        self.data = data
        self.index = index
        self.classes = classes
        self.features = []
        self.classifier = KNeighborsClassifier(5,2, verbose=True)
        self.verbose = True

    # Load mat file
    def load_data(self, file, train=False):
        data_set = loadmat(file, squeeze_me=True)

        self.data = data_set['d']

        # If training, load known index and class data
        if train:        
            self.index = data_set['Index']
            self.classes = data_set['Class']

    # Plots data
    def plot_data(self, x, xlen):
        plt.plot(range(x, x+xlen), self.data[x:x+xlen])
        plt.show()

    # Sort index and class data by index
    def sort(self):        
        sort_zip = sorted(zip(self.index, self.classes))
        self.index, self.classes = map(np.array, zip(*sort_zip))

    #import numpy as np

    def spine_interpolation(self):
        # y is the signal to be interpolated
        
        # Find the indices of the local maxima of the signal
        max_indices = np.argpartition(self.data, -2)[-2:]
        max_indices = max_indices[np.argsort(self.data[max_indices])][::-1]
        
        # Interpolate the signal using the local maxima
        x = np.arange(len(self.data))
        interp = np.interp(x, max_indices, self.data[max_indices])
        self.data = interp
        #return interp


    def savitzky_golay_filter(self, window_size, polynomial_order):
        """Apply a Savitzky-Golay filter to the given data.
        
        Arguments:
            data: The data to be filtered, as a 1-D array.
            window_size: The size of the moving window for the filter.
            polynomial_order: The order of the polynomial used in the filter.
        
        Returns:
            The filtered data, as a 1-D array.
        """
        # apply the Savitzky-Golay filter to the data
        filtered_data = savgol_filter(self.data, window_size, polynomial_order)
        self.data = filtered_data
        #return filtered_data

    # Define the peak enhancement function
    def enhance_peaks(self):
        # Apply the logarithmic transformation to the signal
        enhanced_signal = np.log(self.data)

        self.data = enhanced_signal
    
    # Apply filter to recording data
    def filter(self, cutoff, type, fs=25e3, order=2):
        # Low pass butterworth filter
        sos = butter(order, cutoff, btype=type, output='sos', fs=fs)
        filtered = sosfiltfilt(sos, self.data)
        self.data = filtered   

    # Create window 
    def create_window(self, window_size=46, offset = 15):
        windows = np.zeros((len(self.index), window_size))
        
        # Loop through each spike index
        for i, index in enumerate(self.index):       

            # Slice data array to get window     
            data = self.data[int(index-offset):int(index+window_size-offset)]

            windows[i, :] = data
        return windows
    
    # essentially prominance 
    def adaptive_threshold(self, window_size = 10):
        noise_est = []
        clean_indices = []

        for i in range(len(self.data)):
            ratio = self.data[i]/self.data[i+window_size]
            if ratio < 1 and np.abs(self.data[i+window_size]) > 2:
                clean_indices.append(i)
            
            if len(clean_indices) == 250:
                for i in clean_indices:
                    window = self.data[i:i+window_size]
                    noise_est.append(np.abs(statistics.variance(window)))
                break

        threshold = np.median(noise_est) 
        # print(threshold)
        return (threshold)

    # Define the peak detection function
    def detect_peaks(self):
        # Initialize empty list to store the detected peaks
        peaks = []
        threshold = self.adaptive_threshold()
        
        # Loop through the signal and detect peaks
        for i in range(1, len(self.data) - 1):
            # Check if the current value is a peak
            if self.data[i] > threshold and self.data[i] > self.data[i - 1] and self.data[i] > self.data[i + 1]:
                peaks.append(i)
        
        self.index = peaks
        return peaks

    def find_spikes(self, roll_window = 500, rolling = 1):
        #self.plot_data(0, len(self.data))
        #hold_data = self.data.copy()
        #if rolling == 1:
        #    sig = pd.DataFrame(self.data)
        #    median = sig.rolling(roll_window, min_periods=1).median().to_numpy()
        #    for i in range(len(self.data)):
        #        self.data[i] = self.data[i] - median[i][0]

        #    self.plot_data(0, len(self.data))
        #    threshold = self.adaptive_threshold()
            #threshold = 5 * np.median(np.abs(self.data)/0.6745)
            #print(threshold)
        #    peaks, _ = find_peaks(self.data, height = threshold )

        #    self.index = peaks
            #print(len(peaks))
        #else: 
        threshold = self.adaptive_threshold()
        
        peaks, _ = find_peaks(self.data , height = threshold)
        self.index = peaks

        return peaks

    # Define the feature extraction function
    def extract_features(self): #signal, peaks):
        # Initialize empty list to store the extracted features
        #features = []
        
        # Loop through the detected peaks
        for peak in self.index:
            # Extract the amplitude of the peak
            amplitude = self.data[peak]
            
            # Extract the width of the peak
            left_width = peak
            while left_width > 0 and self.data[left_width] > self.data[left_width - 1]:
                left_width -= 1
            right_width = peak
            while right_width < len(self.data) - 1 and self.data[right_width] > self.data[right_width + 1]:
                right_width += 1
            width = right_width - left_width
            
            # Extract the shape of the peak
            shape = self.data[left_width:right_width + 1]
            
            # Store the extracted features in a list
            self.features.append([amplitude, width, shape])
        
        #return features

    # Define the peak classification function
    def classify_peaks(self): #signal, peaks):
        # Extract features from the signal and peaks
        self.extract_features()
        
        # Use K-Nearest Neighbors classifier to classify the peaks
        
        self.classifier.fit(self.features, self.classes)
        
        # Predict the classes for the detected peaks
        peak_classes = self.classifier.predict(self.index)
        
        return peak_classes


    def cross_validate(self, num_folds=5):
        # Split the data into folds
        kf = KFold(n_splits=num_folds)

        # Initialize a list to store the classifier's performance on each fold
        scores = []

        # Loop through the folds
        for train_index, test_index in kf.split(self.data):
            # Split the data into training and test sets
            X_train, X_test = self.data[train_index], self.data[test_index]
            y_train, y_test = self.classes[train_index], self.classes[test_index]

            # Train the classifier on the training set
            self.classifier.fit(X_train, y_train)

            # Evaluate the classifier's performance on the test set
            score = self.classifier.score(X_test, y_test)
            scores.append(score)

        # Calculate the mean and standard deviation of the classifier's performance across the folds
        mean_score = np.mean(scores)
        std_dev = np.std(scores)

        return mean_score, std_dev

    def train_classifier(classifier, X, y):
        # Train the classifier on the data
        classifier.fit(X, y)

        # Return the trained classifier
        return classifier