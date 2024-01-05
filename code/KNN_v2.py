import numpy as np
from scipy.io import savemat

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier

from spike_functions_v2 import Spike_object

class Spike_sort_KNN:

    def __init__(self, k=5, p=2, verbose=True):        
  
       self.n = KNeighborsClassifier(n_neighbors=k, p=p)

       self.verbose = verbose
        

    def train_knn(self):       

        # Create SpikeData object for training data
        self.training_data = Spike_object()

        # Load in the training data set
        #self.training_data.load_mat('training.mat', train=True)  
        self.training_data.load_data('D1.mat', train=True)  
        print(len(self.training_data.index))
        #self.training_data.plot_data(0,1440000)
        self.training_data.plot_data(1062, 500)

        # Sort index/class data 
        self.training_data.sort()

        # Take last 20% of training data set and use as a validation data set
        self.validation_data = self.training_data.split(0.2)

        # Filter the raw data
        #self.training_data.data = pywt.dwt(self.training_data.data, 'db1')
        self.training_data.savitzky_golay_filter(25, 5)
        #self.training_data.spine_interpolation()
        #self.training_data.plot_data(0,len(self.training_data.data))
        #self.training_data.plot_data(1062, 500)

        #self.training_data.filter(2500, 'low') 
        #self.training_data.plot_data(1062, 500)
        #self.validation_data.filter(2500, 'low')
        self.validation_data.savitzky_golay_filter(25, 5)

        # Run spike detection and comparison on training data
        self.training_data.compare_spikes()

        # Train the MLP with training dataset classes        
        self.n.fit(self.training_data.create_window(), self.training_data.classes)


    def validate_knn(self):     
        # Run spike detection and comparison on validation data
        spike_score = self.validation_data.compare_spikes()  

        # Classify detected spikes
        predicted = self.n.predict(self.validation_data.create_window())
        # Convert probabilties back to class labels (1-5)
        #print(predicted)
        # Compare to known classes
        classified = np.where(predicted == self.validation_data.classes)[0]

        # Score classifier method
        class_score = (len(classified) / len(self.validation_data.index))

        #Performance metrics
        if self.verbose:
            print(f'Spike detection score: {spike_score:.4f}')
            print(f'Class detection score: {class_score:.4f}')
            print(f'Overall score:{(spike_score*class_score):.4f}')

            cm = confusion_matrix(self.validation_data.classes, predicted)
            print(cm)
            cr = classification_report(self.validation_data.classes, predicted, digits=4)
            print(cr)

        return class_score


    # Run classifier on submission data set and create submission file
    def submission(self):
        self.submission_data = Spike_object()
        #self.submission_data.load_mat('submission.mat')
        self.submission_data.load_data('D1.mat', train=False)

        # Filter data with band pass as data is very noisy
        self.submission_data.filter([25,1800], 'band')
        #self.submission_data.filter(3200, 'low')

        spikes = self.submission_data.find_spikes()
        print(f'{len(spikes)} spikes detected')
        #self.submission_data.plot_data(1062,500)

        predicted = self.n.predict(self.submission_data.create_window())
        self.submission_data.classes = predicted      

        print('Class Breakdown')
        self.class_breakdown(predicted)

        mat_file = {'Index': self.submission_data.index, 'Class':predicted}
        savemat('D2_output.mat', mat_file)


    def class_breakdown(self, classes):
        unique, counts = np.unique(classes, return_counts=True)
        breakdown = dict(zip(unique, counts))

        for key, val in breakdown.items():
            print(f'Type {key:g}: {val}')
        return breakdown


if __name__ == '__main__':
    ####### task 1 ######
    s = Spike_sort_KNN(5,2, verbose=True)
    s.train_knn()
    s.validate_knn()
    s.submission()