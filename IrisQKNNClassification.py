import numpy as np
import qiskit as qk
from qiskit.utils import QuantumInstance
from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
from skimage.measure import label, regionprops_table
import numpy as np
import cv2
import glob
import time
import pandas as pd
import math
from PIL import Image
from pathlib import Path
from sklearn import metrics, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Variable
TrainPath = "Data Pengenalan/train"
ValPath = "Data Pengenalan/val"

#Creating QKNN Section
Shot = 10000 #number of iteration that quantum execute
Neighbors = 3 #need to be optimize

#Input data 
Na_Train = 0
Nb_Train = 127   # max 127
Na_val = 0
Nb_val = 23  # max 23

standarisasi = False

Pengulangan = 2

def ChangeFormat(PATH_START,PATH_GOAL):
    # PATH = "data uji"
    files = glob.glob (str(PATH_START)+"/*.gif")
    print(files)
    image_array = []

    for myFile in files:
        img = Image.open(myFile)
        files_name = (Path(myFile).stem)
        new_name = f'{PATH_GOAL}/{files_name}.jpg'
        img.convert('RGB').save(new_name)
    print('The Image has changed')

def read_image(PATH):
    files = glob.glob (str(PATH)+"/*.jpg")

    # print(files)
    print("Jumlah files sebanyak", len(files))
    image_array = []
    Files_Name = []
    #Loop to store the mask to mask array
    for myFile in files:
        img = cv2.imread (myFile,0) #0 untuk gray form, sedangkan 1 untuk color form1
        (thresh, BWmask) = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY) #Store them in Black and White 0-1 form 
        image_array.append (BWmask) #append each image to array
        file_name = (Path(myFile).stem)
        Files_Name.append(file_name)

    return image_array, Files_Name

def ExtractName2Class(FileNames):
    Class = []
    for number in range(len(FileNames)):
        Class.append(FileNames[number].split('-')[0])
    print('Class has been extracted')
    return Class

def ExtractProperties(Images):
     
    Number_Image = len(Images)
    print(f"There are {Number_Image} images") 
    AllRegions = []
    for number in range(Number_Image):
        img = Images[number]
        label_img = label(img)
        regions = regionprops_table(label_img, img, properties = ['area', 'perimeter', 'eccentricity' ])
        
        #Extract to handle eror area
        regions['area'] = regions['area'][0]
        regions['perimeter'] = regions['perimeter'][0]
        regions['eccentricity'] = regions['eccentricity'][0]
        AllRegions.append(regions)
    print("Region has got!")
    return AllRegions
        # print(TrainImage[number], TrainName[number])

def CalcCircularity(data):
    #Get The Circularity Value
    area = data['area']
    perimeter = data['perimeter']
    circularity = 4*math.pi*area/np.power(perimeter,2)
    data.insert(4,"Circularity",circularity, False)

def encode_pad(data):

    # Change the dtype of data to be float for safety encoding
    data = data.astype(float)
    # Encoding
    encoded_data = analog.encode(data)

    # Padding data to be 2^n dimensional with n=2
    encoded_data = np.pad(encoded_data,
                          pad_width=((0, 0), (0, 1)),
                          mode='constant',
                          constant_values=(0))
    print("Shape after encoded and paded ", encoded_data.shape)
    return encoded_data

iris = datasets.load_iris()
# Create a dataframe
X = iris.data[:, :4]
y = iris.target

x_train, x_valid, train_label, valid_label = train_test_split( X, y, test_size=0.15, random_state=14)

#Standarisasi
if standarisasi == True:
    print('Do Standardization')
    scaler = StandardScaler()
    model1 = scaler.fit(x_train)
    scaled_train_data = model1.transform(x_train)

    model2 = scaler.fit(x_valid)
    scaled_valid_data = model2.transform(x_valid)

    #Encoding
    encoded_x_train = analog.encode(scaled_train_data)
    encoded_x_valid = analog.encode(scaled_valid_data)
else:
    print("Do Encode only")    
    #Encoding
    encoded_x_train = analog.encode(x_train)
    encoded_x_valid = analog.encode(x_valid)

#fixing the passing dataset
encoded_x_train = encoded_x_train[Na_Train:Nb_Train]
encoded_x_valid = encoded_x_valid[Na_val:Nb_val]
train_label = train_label[Na_Train:Nb_Train]
valid_label = valid_label[Na_val:Nb_val]

for ulang in range(Pengulangan):
    print(f"\n\n ====== Percobaan ke-{ulang} dari {Pengulangan} sedang dilakukan ======= \n\n")

    #QKNN Section
    txt = f"Creating Quantum Circuit with {Neighbors} neighbors and {Shot} shots"
    x = txt.center(30, "=")
    print(x)
    backend = qk.BasicAer.get_backend('qasm_simulator')
    instance = QuantumInstance(backend, shots= Shot)
    qknn = QKNeighborsClassifier(
        n_neighbors= Neighbors,
        quantum_instance=instance
    )

    Start = time.time()
    AutoCircuit = qknn.construct_circuits(
        data_to_predict=encoded_x_valid, training_data=encoded_x_train)
    end = time.time() - Start
    print("Time for creating the circuits", end)

    AutoCircuitResult = qknn.get_circuit_results(AutoCircuit)
    Auto_all_counts = AutoCircuitResult.get_counts()
    MyAutoFidelities = qknn.get_all_fidelities(AutoCircuitResult)

    qknn_prediction = qknn.majority_vote(
        labels=train_label,
        fidelities=MyAutoFidelities
    )

    print("Nilai Fidelities",MyAutoFidelities[0])

    from qiskit.visualization import plot_histogram
    Auto_all_counts = AutoCircuitResult.get_counts()
    # print(Auto_all_counts)
    plot_histogram(Auto_all_counts[0], figsize=(20, 5))

    Start = time.time()
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = Neighbors)
    # Fit the classifier to the data
    knn.fit(encoded_x_train,train_label)
    knn_prediction = knn.predict(encoded_x_valid)
    end = time.time()-Start
    print(end,  'second, waktu predict dan fit Classical KNN')

    txt = "Quantum k-NN Result"
    x = txt.center(30, "=")
    print(x)
    print("Predicted label : ",qknn_prediction)
    print("Real label is", valid_label)
    qknn_acc = metrics.accuracy_score(valid_label, qknn_prediction)
    qknn_confusion_matrix = metrics.confusion_matrix(valid_label, qknn_prediction)
    print(qknn_acc, qknn_confusion_matrix)

    txt = "Clasik k-NN Result"
    x = txt.center(30, "=")
    print(x)
    print("Predicted label : ",knn_prediction)
    print("Real label is", valid_label)
    knn_acc = metrics.accuracy_score(valid_label, knn_prediction)
    knn_confusion_matrix = metrics.confusion_matrix(valid_label, knn_prediction)
    print(knn_acc, knn_confusion_matrix)