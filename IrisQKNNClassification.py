import numpy as np
import qiskit as qk
from qiskit.utils import QuantumInstance
from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
import numpy as np
import time
import pandas as pd
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
Nb_Train = 32   # max 127
Na_val = 0
Nb_val = 8  # max 23

standarisasi = False

Pengulangan = 1

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