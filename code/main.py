import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.signal import butter, sosfiltfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import copy
import os

def noise_LP_filter(raw_signal, cutoff_freq=2500,sampling_freq=25000,order=2):

    filter=butter(order,cutoff_freq,'low',analog=False,output='sos',fs=sampling_freq)
    filtered_Signal=sosfiltfilt(filter,raw_signal)

    return filtered_Signal

def peak_detection(signal, RM_window, threshold):

    signalDF=pd.DataFrame(signal)

    if RM_window==0:
        normalised_signal=signal.copy()

        MAD=sp.stats.median_abs_deviation(signal)
        median=MAD
        peak_threshold = 5 * median

        peaks, _ = find_peaks(signal, height=peak_threshold)

        median=[MAD]*len(signal)
    else:
        normalised_signal=signal.copy()
        
        median=signalDF.rolling(window=RM_window, min_periods=1).median().to_numpy()

        for i in range(len(signal)): normalised_signal[i]=signal[i]-median[i][0]

        peaks,_ = find_peaks(normalised_signal, height=threshold)

    return peaks,normalised_signal,median

def extract_data(signal, RM_window, spike_window, threshold, index_list=0, class_list=0):

    if type(index_list)!=int:

        spikes_list = [] # Initialize array of sample points
        neuron_class_list = [] # Initialize array of neuron types
        duplicates = [] # Initialize duplicates array
        index_list_ordered=[]
        peaks_list=[]

        peaks,normalised_signal,median=peak_detection(signal,RM_window, threshold)  

        for peak_i in peaks:
            
            i = index_list[index_list < peak_i].max()

            if i in duplicates: continue

            duplicates.append(i)

            n = np.where(index_list==i)[0][0]
            
            values = normalised_signal[peak_i-spike_window[0]:peak_i+spike_window[1]]

            spikes_list.append(values)
            neuron_class_list.append(class_list[n])
            index_list_ordered.append(i)
            peaks_list.append(peak_i)

        peaks_identified=len(peaks_list)/len(index_list)

        return spikes_list, neuron_class_list, median, normalised_signal,index_list_ordered,peaks_list,peaks_identified

    else:
        spikes_list = []
        index_list=[] 
        peaks_list=[]

        peaks,normalised_signal,median=peak_detection(signal,RM_window, threshold)  

        for peak_i in peaks:

            index=peak_i-spike_window[0]
            
            values = normalised_signal[peak_i-spike_window[0]:peak_i+spike_window[1]]

            spikes_list.append(values)
            index_list.append(index)
            peaks_list.append(peak_i)

        return spikes_list, median, normalised_signal, index_list, peaks_list
    
def signal_processing(signal, RM_window, spike_window, cutoff_freq, threshold, Index=0, Class=0):
    
    # Denoise the training recording with a low-pass filter
    filtered_Data = noise_LP_filter(signal, cutoff_freq)

    if type(Class)!=int:

        # Extract the spikes from the training recording and getting the corresponding neuron class values
        spikes,classes,median, normalised_signal,index_sorted,peaks_found,peaks_identified_rate = extract_data(filtered_Data,  RM_window, spike_window, threshold, Index, Class)

        return spikes,classes,median,normalised_signal,index_sorted,peaks_found,peaks_identified_rate

    else:

        spikes,median,normalised_signal,index_list,peaks_list=extract_data(filtered_Data,RM_window,spike_window, threshold)

        return spikes,median, normalised_signal, index_list,peaks_list
   
def find_rise_times(peaks_found,index_sorted,classes):

    rise_times=peaks_found.copy()

    for i in range(len(peaks_found)):
        rise_times[i]=peaks_found[i]-index_sorted[i]

    indices_1 = [i for i, x in enumerate(classes) if x == 1]
    indices_2 = [i for i, x in enumerate(classes) if x == 2]
    indices_3 = [i for i, x in enumerate(classes) if x == 3]
    indices_4 = [i for i, x in enumerate(classes) if x == 4]
    indices_5 = [i for i, x in enumerate(classes) if x == 5]

    ART_Class1=0

    for i in indices_1:
        ART_Class1=ART_Class1+rise_times[i]

    ART_Class1=ART_Class1/len(indices_1)

    ART_Class2=0

    for i in indices_2:
        ART_Class2=ART_Class2+rise_times[i]

    ART_Class2=ART_Class2/len(indices_2)

    ART_Class3=0

    for i in indices_3:
        ART_Class3=ART_Class3+rise_times[i]

    ART_Class3=ART_Class3/len(indices_3)

    ART_Class4=0

    for i in indices_4:
        ART_Class4=ART_Class4+rise_times[i]

    ART_Class4=ART_Class4/len(indices_4)

    ART_Class5=0

    for i in indices_5:
        ART_Class5=ART_Class5+rise_times[i]

    ART_Class5=ART_Class5/len(indices_5)

    Average_Rise_Times=[ART_Class1,ART_Class2,ART_Class3,ART_Class4,ART_Class5]

    return Average_Rise_Times

def find_spike_index(peaks,classes,rise_times):
    
    index_list=[]

    for i in range(len(peaks)):
        
        if classes[i]==1:
            index_list.append(round(peaks[i]-rise_times[0]))
        elif classes[i]==2:
            index_list.append(round(peaks[i]-rise_times[1]))
        elif classes[i]==3:
            index_list.append(round(peaks[i]-rise_times[2]))
        elif classes[i]==4:
            index_list.append(round(peaks[i]-rise_times[3]))
        elif classes[i]==5:
            index_list.append(round(peaks[i]-rise_times[4]))
        
    return index_list


# /////////// DATASET 1 ////////////////

#/////////// OPEN DATASET 1
os.chdir('Coursework_C/Data_Sets')

training_data_file='D1.mat'

training_path = training_data_file
 
data_file_train = spio.loadmat(training_path, squeeze_me=True)

Index = data_file_train['Index'] #  The location in the recording (in samples) of each spike.
Class = data_file_train['Class'] # The class (1, 2, 3 or 4), i.e the type of neuron that generated each spike.
Training_Data = data_file_train['d'] # Raw time domain recording for training dataset


#\\\\\\\\\\\\ BUILD NETWORK
k = 5 
p = 2 

model = KNeighborsClassifier(n_neighbors=k, p=p)


# /////// TRAIN AND TEST NETWORK
RM_window=500
spike_window=[15,26]
cutoff_freq=2500
threshold=1.069
# threshold=0.7

training_spikes,classes,median1,normalised_signal1,index_sorted,peaks_found,peaks_rate = signal_processing(Training_Data,RM_window,spike_window,cutoff_freq,threshold,Index,Class)

print(peaks_rate)

# Find Average Rise Time for each class of neuron
Average_Rise_Times=find_rise_times(peaks_found,index_sorted,classes)

# Split training dataset with an 80-20 split
train_spikes, test_spikes, train_class, test_class = train_test_split(training_spikes, classes, test_size=0.2)

model.fit(train_spikes, train_class)

print('Finished Training')

# Predict classes for spikes in the test subset
predicted_class = model.predict(test_spikes)

# Display confusion matrix
confusion_matrix = metrics.confusion_matrix(test_class, predicted_class)
print(confusion_matrix)

# Display performance metrics
performance_metrics = metrics.classification_report(test_class, predicted_class, digits=4)
print(performance_metrics)

print('Finished Testing')





# ////////////// DATASET 2 ////////////////

# ///////////////////// OPEN DATASET 2
Dataset2_filename='D2.mat'

Dataset2_file = spio.loadmat(Dataset2_filename, squeeze_me=True)

Data_2 = Dataset2_file['d'] 

#//////////////////////Extract Data to classify
spikes_to_classify,median2,normalised_signal2, index_list_unclassified, unclassified_peaks=signal_processing(Data_2,RM_window,spike_window,cutoff_freq,threshold)

# ////////////////////// Classify data
predicted_classes_dataset2 = model.predict(spikes_to_classify)

# /////////////////////// find predicted spike indeces
predicted_indices_dataset2=find_spike_index(unclassified_peaks,predicted_classes_dataset2,Average_Rise_Times)

print('Finished Datset 2')




# /////////////// DATASET 3 ////////////////
Dataset3_filename='D3.mat'

Dataset3_file = spio.loadmat(Dataset3_filename, squeeze_me=True)

Data_3 = Dataset3_file['d'] 

#//////////////////////Extract Data to classify
spikes_to_classify,median3,normalised_signal3, index_list_unclassified, unclassified_peaks=signal_processing(Data_3,RM_window,spike_window,cutoff_freq,threshold)

# ////////////////////// Classify data
predicted_classes_dataset3 = model.predict(spikes_to_classify)

# /////////////////////// find predicted spike indeces
predicted_indices_dataset3=find_spike_index(unclassified_peaks,predicted_classes_dataset3,Average_Rise_Times)

print('Finished Datset 3')




# /////////////// DATASET 4 ////////////////
Dataset4_filename='D4.mat'

Dataset4_file = spio.loadmat(Dataset4_filename, squeeze_me=True)

Data_4 = Dataset4_file['d'] 

#//////////////////////Extract Data to classify
spikes_to_classify,median4,normalised_signal4, index_list_unclassified, unclassified_peaks=signal_processing(Data_4,RM_window,spike_window,cutoff_freq,threshold)

# ////////////////////// Classify data
predicted_classes_dataset4 = model.predict(spikes_to_classify)

# /////////////////////// find predicted spike indeces
predicted_indices_dataset4=find_spike_index(unclassified_peaks,predicted_classes_dataset4,Average_Rise_Times)

print('Finished Datset 4')





# //////////// SAVE RESULTS TO .MAT //////////////////

os.chdir('..')
os.chdir('Matlab_Results')

folder_name = 'Classification_Results'
file_count = 0
folder_path = folder_name + str(file_count)

while os.path.isdir(folder_path):
    file_count += 1
    folder_path = folder_name + '_' + str(file_count)

os.mkdir(folder_path)
os.chdir(folder_path)

# TASK 1 /////////////

folder='Task_1'
file_name='D2.mat'

os.mkdir(folder)
os.chdir(folder)

dic_Dataset2 = {"D": Data_2.tolist(),"Index": predicted_indices_dataset2}

spio.savemat(file_name, dic_Dataset2)

os.chdir('..')

# TASK 2 /////////////

folder='Task_2'
file_name='D2.mat'

os.mkdir(folder)
os.chdir(folder)

dic_Dataset2 = {"D": Data_2.tolist(),"Index": predicted_indices_dataset2, "Class": predicted_classes_dataset2.tolist()}

spio.savemat(file_name, dic_Dataset2)

os.chdir('..')

# TASK 3 ////////////

folder='Task_3'
file_name='D3.mat'

os.mkdir(folder)
os.chdir(folder)

dic_Dataset3 = {"D": Data_3.tolist(),"Index": predicted_indices_dataset3, "Class": predicted_classes_dataset3.tolist()}

spio.savemat(file_name, dic_Dataset3)

os.chdir('..')

# TASK 4 ////////////

folder='Task_4'
file_name='D4.mat'

os.mkdir(folder)
os.chdir(folder)

dic_Dataset4 = {"D": Data_4.tolist(),"Index": predicted_indices_dataset4, "Class": predicted_classes_dataset4.tolist()}

spio.savemat(file_name, dic_Dataset4)

os.chdir('..')


#////////////////////////// Plot Results (Optional)
# Filtered Data vs median value
# plt.plot(normalised_signal1)
# plt.plot(median1)
# plt.show()
# plt.plot(normalised_signal2)
# plt.plot(median2)
# plt.show()

print('done babayyyy')