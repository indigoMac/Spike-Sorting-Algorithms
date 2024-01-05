Required packages:
- scipy.signal
- scipy.io
- numpy 
- matplotlib.pyplot
- statistics
- scipy.optimize
- scipy.fft
- pandas
- sklearn.model_selection
- sklearn.metrics.pairwise
- os
- sklearn.neighbors
- sklearn.metrics

Python files:
- KNN_main.py
- spike_functions.py

All training data and testing data (D1.py, D2.py, D3.py, D4.py)
must be in the same directory as the two python files.
Running the KNN_main.py file will run all tests and produce .mat file 
outputs to a directory named 'ResultsN' where N is a number.

By chanign the 'file_num' input to the 'Spike_sort_KNN.output()' spike_function
to change the method of signal processing prior to the classification.

file_num = 1 is for dual annealling
file_num = 2 is for the savitzky_golay_filter (used on D2.mat)
file_num = 3 is for the bandpass filter with fixed cutoff frequencies (used for D3.mat and D4.mat)
