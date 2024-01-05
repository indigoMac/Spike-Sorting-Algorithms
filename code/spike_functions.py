from scipy.signal import find_peaks, butter, sosfiltfilt, savgol_filter, lfilter
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import statistics
import scipy.optimize._dual_annealing
from scipy.fft import fft, ifft
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import euclidean_distances

class Spike_object:
    """
    A class to store and manipulate spike data, as well as spike index and class data.
    """

    # set variables
    def __init__(self, data=None, index=None, classes=None):
        """
        Initializes the SpikeObject with optional data, index, and classes.
        
        Parameters:
        data (np.array): The raw spike data.
        index (np.array): Indices of spikes in the data.
        classes (np.array): Classification of each spike.
        """
        self.data = data
        self.index = index
        self.classes = classes
        #self.

    # Load mat file
    def load_data(self, file, train=False):
        """
        Loads data from a .mat file.

        Parameters:
        file (str): Path to the .mat file.
        train (bool): If true, also loads index and class data for training.

        Raises:
        FileNotFoundError: If the file cannot be found.
        KeyError: If expected data keys are not in the file.
        """
        try:
            data_set = loadmat(file, squeeze_me=True)
            self.data = data_set['d']
            if train:        
                self.index = data_set['Index']
                self.classes = data_set['Class']
        except FileNotFoundError:
            raise FileNotFoundError(f"File {file} not found.")
        except KeyError as e:
            raise KeyError(f"Missing expected data key in file: {e}")

    # Plots data
    def plot_data(self, x, xlen):
        """
        Plots a segment of the spike data.

        Parameters:
        x (int): The starting index of the segment to be plotted.
        xlen (int): The length of the segment to be plotted.
        """
        plt.plot(range(x, x+xlen), self.data[x:x+xlen])
        plt.show()

        if x + xlen > len(self.data):
            xlen = len(self.data) - x  # Adjust length if it exceeds data bounds
        plt.plot(range(x, x + xlen), self.data[x:x + xlen])
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.title('Spike Data Segment')
        plt.show()

    # Sort index and class data by index
    def sort(self):        
        """
        Sorts the index and class data by index.
        
        Precondition: Both self.index and self.classes must be non-empty and of the same length.
        """
        if self.index is None or self.classes is None:
            print("Index or class data is missing.")
            return
        if len(self.index) != len(self.classes):
            print("Index and class data length mismatch.")
            return
        sort_zip = sorted(zip(self.index, self.classes))
        self.index, self.classes = map(np.array, zip(*sort_zip))

    # instead of filtering use a spine interpolation function
    def spine_interpolation(self):
        """
        Applies spine interpolation to the data.

        This method finds the local maxima of the signal and interpolates the signal using these points.
        """
        if self.data is None:
            print("Data is not loaded.")
            return
        # Find the indices of the local maxima of the signal
        max_indices = np.argpartition(self.data, -2)[-2:]
        max_indices = max_indices[np.argsort(self.data[max_indices])][::-1]
        #max_indices.sort()
        
        # Interpolate the signal using the local maxima
        x = np.arange(len(self.data))
        interp = np.interp(x, max_indices, self.data[max_indices])
        self.data = interp


    # savitzky_golay_filter (output similar to low pass filter)
    def savitzky_golay_filter(self, window_size, polynomial_order):
        """
        Applies the Savitzky-Golay filter to smooth the data.

        Parameters:
        window_size (int): The size of the filter window (must be odd and greater than polynomial_order).
        polynomial_order (int): The order of the polynomial used to fit the samples.
        """
        if self.data is None:
            print("Data is not loaded.")
            return
        
        self.data = savgol_filter(self.data, window_size, polynomial_order)
        
        
    # bandpass filter using the fourier transform
    def bandpass_filter(self, min_frequency, max_frequency):
        """
        Applies a bandpass filter using the Fourier transform.

        Parameters:
        min_frequency (int): The minimum frequency to pass.
        max_frequency (int): The maximum frequency to pass.

        Returns:
        np.array: The filtered signal.
        """
        if self.data is None:
            print("Data is not loaded.")
            return None
        # Apply a Fourier transform to the signal to convert it to the frequency domain
        frequency_spectrum = fft(self.data)

        # Create a mask that will only keep the frequencies within the specified range
        mask = np.zeros(len(frequency_spectrum))
        mask[min_frequency:max_frequency] = 1

        # Apply the mask to the frequency spectrum to filter the signal
        filtered_spectrum = frequency_spectrum * mask
        return ifft(filtered_spectrum)

    # specilaised bandpass filter function
    def BP_filter(self, low_cut,high_cut,sampling_freq,order=2):
        """
        Applies a specialized bandpass filter to the data.

        Parameters:
        low_cut (float): The lower frequency bound of the filter.
        high_cut (float): The upper frequency bound of the filter.
        sampling_freq (float): The sampling frequency of the data.
        order (int): The order of the filter.

        Returns:
        np.array: The filtered data.
        """
        if self.data is None:
            print("Data is not loaded.")
            return None
        nyq = 0.5 * sampling_freq
        low = low_cut / nyq
        high = high_cut / nyq
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, self.data)
    
    # Apply filter to recording data
    def filter(self, cutoff, type, fs=25e3, order=2):
        """
        Applies a low-pass or high-pass Butterworth filter to the data.

        Parameters:
        cutoff (float): The cutoff frequency of the filter.
        filter_type (str): Type of the filter ('low' or 'high').
        fs (float): The sampling frequency of the data.
        order (int): The order of the filter.
        """
        if self.data is None:
            print("Data is not loaded.")
            return
        # Low pass butterworth filter
        sos = butter(order, cutoff, btype=type, output='sos', fs=fs)
        self.data = sosfiltfilt(sos, self.data) 

    # Create window for class identification
    def create_window(self, window_size=46, offset = 15):
        """
        Creates windows around each spike index for class identification.

        Parameters:
        window_size (int): The size of each window.
        offset (int): The offset from the spike index to start the window.

        Returns:
        np.array: An array of windows for classification.
        """
        if self.index is None or len(self.index) == 0:
            print("Index data is not available.")
            return None
        windows = np.zeros((len(self.index), window_size))
        
        # Loop through each spike index
        for i, index in enumerate(self.index):       
            # Slice data array to get window     
            start = int(max(index - offset, 0))
            end = int(min(index + window_size - offset, len(self.data)))
            windows[i, :end - start] = self.data[start:end]
        return windows
    
    # measures the averarge noise in a window and 
    # returns a threshold depenmding on this
    def adaptive_threshold(self, window_size = 10):
        """
        Calculates an adaptive threshold based on the variance within windows of the data.

        Parameters:
        window_size (int): The size of the window to estimate noise.

        Returns:
        float: The calculated threshold value.
        """
        if self.data is None:
            print("Data is not loaded.")
            return None
        noise_est = []
        for i in range(len(self.data) - window_size):
            window = self.data[i:i + window_size]
            if np.ptp(window) > 2:  # Using peak-to-peak to identify significant windows
                noise_est.append(np.var(window))
        return np.median(noise_est) if noise_est else 0

    # get rolling medium of signal and subtract it to faltten signal, bandpass filter works better
    def messy_signal(self, RM_window=500):
        """
        Applies a rolling median subtraction to the data to flatten the signal.

        Parameters:
        RM_window (int): The window size for calculating the rolling median.
        """
        if self.data is None:
            print("Data is not loaded.")
            return
        median = pd.Series(self.data).rolling(window=RM_window, min_periods=1).median().to_numpy()
        self.data -= median.ravel()
        self.plot_data(0, len(self.data))
    
    # find spikes using prominance
    def find_spikes(self, low_cut=0, high_cut=0, xprominence =5, wlen=30, sampling_freq=25e6):
        """
        Identifies spikes in the data using prominence.

        Parameters:
        low_cut (float): Lower frequency bound for optional bandpass filter.
        high_cut (float): Upper frequency bound for optional bandpass filter.
        xprominence (float): Multiplier for the prominence threshold.
        wlen (int): Window length for peak finding.
        sampling_freq (float): Sampling frequency of the data.

        Returns:
        np.array: Indices of identified spikes.
        """
        if self.data is None:
            print("Data is not loaded.")
            return None
        if low_cut != 0 and high_cut != 0:
            self.data = self.BP_filter(low_cut, high_cut, sampling_freq)
        MAD = scipy.stats.median_abs_deviation(self.data)
        peaks, _ = find_peaks(self.data, prominence=xprominence * MAD, wlen=wlen)
        self.index = peaks
        return peaks

    # find peaks using an amplitude threshold
    def find_spike(self):
        """
        Identifies spikes in the data using an amplitude threshold.

        Returns:
        np.array: Indices of identified spikes.
        """
        if self.data is None:
            print("Data is not loaded.")
            return None
        threshold = self.adaptive_threshold()
        peaks, _ = find_peaks(self.data, height=threshold)
        self.index = peaks
        return peaks


    # Define the peak detection function
    def detect_peaks(self, xprominence, wlen):
        """
        Detects peaks in the data based on prominence.

        Parameters:
        xprominence (float): Multiplier for the prominence threshold.
        wlen (int): Window length for peak finding.

        Returns:
        np.array: Indices of detected peaks.
        """
        MAD=scipy.stats.median_abs_deviation(self.data)

        peaks,_ = find_peaks(self.data, prominence=xprominence*MAD, wlen=wlen)

        return peaks

    # function to prounces the peaks further for better peak identification
    def enhance_peaks(self):
        """
        Enhances the peaks in the data for better identification.

        Applies a logarithmic transformation to the data.
        """
        if self.data is None:
            print("Data is not loaded.")
            return
        self.data = np.log(self.data)

    # Spilt one spike data set into 2 (training/validation)
    def split_data(self, percent):
        """
        Splits the data into two sets based on the given percentage.

        Parameters:
        percent (float): The percentage of data to be included in the second set.

        Returns:
        SpikeObject: A new SpikeObject containing the second set of data.
        """
        # Get index to split at
        index = int(len(self.data)*(1.0-percent))
        # Slice data
        split_data = self.data[index:]
        self.data = self.data[:index]

        # Slice spike and class index data
        i = int(np.argmax(self.index >= index))        
        split_spikes = self.index[i:] - index
        self.index = self.index[:i]
        split_classes = self.classes[i:]
        self.classes = self.classes[:i]        

        # Construct new Spike_object object with sliced data and return
        return Spike_object(split_data, split_spikes, split_classes)

    
    # If data is training data, compare found spikes with known index/class 
    def compare(self, range=50):
        """
        Compares found spikes with known index/class data to calculate a score.

        Parameters:
        range (int): The range within which a spike is considered correctly identified.

        Returns:
        float: The score of spike detection.
        """
        if self.index is None or self.classes is None:
            print("Index or class data is missing.")
            return 0
        
        # Get spikes from training data
        known_index = self.index

        # find spikes in data
        found_index = self.find_spike()
       
        # create empty arrays with corrrect sizes
        spikes = np.zeros(len(found_index))
        classes = np.zeros(len(found_index))

        # Loop through found indices
        for i, spike in enumerate(found_index):
            # find where found spike is in know spike array
            # confirm if found spikes are approximatly correct (within range of 50 above or below)
            found = np.where((known_index > spike - range) & (known_index < spike + range))[0]
            
            # if a spike is confirmed 
            if len(found) > 0:

                # Mark as correct in spike array
                spikes[i] = spike
                
                # find difference between where the found spike is and its known position
                diff = abs(known_index[found] - spike)

                # get index of the min corresponding spike
                index = found[np.argmin(diff)]

                # if index exists in self.classes
                if index < len(self.classes):
                    # assign class to be same as the peak found
                    classes[i] = self.classes[index]

                # Remove from selection to prevent repeats
                known_index[index] = 0

        # remove all undetected spikes and the corresponding class from the array
        spikes = spikes[spikes != 0]
        classes = classes[classes != 0]

        # overwrite 
        self.index = spikes
        self.classes = classes

        return (len(spikes) / len(known_index))
    
    # secondary method for comparing found peaks to known peaks
    def compare_peaks(self, classifier, range=50):
        """
        Compares found peaks with known peaks using a classifier.

        Parameters:
        classifier (Classifier): A trained classifier.
        range (int): The range within which a peak is considered correctly identified.

        Returns:
        Tuple[np.array, np.array]: Arrays of correctly identified peaks and their classes.
        """
        if self.index is None:
            print("Index data is missing.")
            return None, None
        
        # Get spikes from training data
        known_peaks = self.index

        # find spikes in data
        found_peaks = self.find_spikes()
        #found_peaks = self.detect_peaks()
        
        # Classify the found peaks using the trained classifier
        found_classes = classifier.predict(found_peaks)

        # Calculate the Euclidean distances between the found and known peaks
        distances = euclidean_distances(found_peaks, known_peaks)

        # Find the minimum distance for each found peak
        # Determine which found peaks have a minimum distance below the threshold
        below_threshold = np.min(distances, axis=1) < range

        # Remove the found peaks and their corresponding classes that do not have a minimum distance below the threshold

        return found_peaks[below_threshold], found_classes[below_threshold]

    # method to verify the performance of the KNN
    def cross_validate(self, classifier, num_folds=5):
        """
        Performs cross-validation on the classifier with the data.

        Parameters:
        classifier (Classifier): The classifier to be evaluated.
        num_folds (int): The number of folds for cross-validation.

        Returns:
        Tuple[float, float]: The mean score and standard deviation of the classifier's performance.
        """
        if self.data is None or self.classes is None:
            print("Data or class data is missing.")
            return 0, 0
        
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
            classifier.fit(X_train, y_train)

            # Evaluate the classifier's performance on the test set
            score = classifier.score(X_test, y_test)
            scores.append(score)

        # Calculate the mean and standard deviation of the classifier's performance across the folds
        mean_score = np.mean(scores)
        std_dev = np.std(scores)

        return mean_score, std_dev
