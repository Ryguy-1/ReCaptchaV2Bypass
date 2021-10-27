# python 3.7
# Scikit-learn ver. 0.23.2
from imblearn import pipeline
from scipy.sparse import data
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
# Imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
# OpenCv
import cv2
# matplotlib 3.3.1
from matplotlib import pyplot
# Numpy
import numpy as np
# DataLoader
from DataLoader import Dataset
# from DataLoader import MiniBatch
# Pickle
import pickle


# Hyperparameters
epochs = 20
batch_size = 64
max_each_class = 1000

def main():
    train()

def train():
    
    data_loader = Dataset('Recaptcha_Data_2\\', batch_size=batch_size, max_each_class=1000)
    # Verify Sizes
    print(f'Data Loader Length: {len(data_loader)}')
    # Declare Model
    model = SGDClassifier(random_state=0, loss='hinge', penalty='l2') # verbose=1
    # Train Loop
    # Set Rounds Per Epoch
    rounds_per_epoch = int(len(data_loader)/batch_size)
    # Test Data
    test_x, test_y = data_loader.get_test_data()

    for epoch in range(epochs):
        for round in range(rounds_per_epoch):
            data, label = data_loader.get_next_batch()
            model.partial_fit(data, label, classes=data_loader.num_classes_list)
            if round % 15 == 0:
                print(f'Batches Checked: {round}/{rounds_per_epoch}')
        print(f'Epoch: {epoch}/{epochs}')
        test(model, test_x, test_y)
        data_loader.reset_index()
        plot_confusion_matrix(model, test_x, test_y)

    for i in range(len(test_y)):
        check_image(test_x[i], test_y[i], model)


def test(model, test_x_final, test_y_final):
    preds = model.predict(test_x_final)
    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, test_y_final):
        if pred == gt: correct += 1
        else: incorrect += 1
    print(f"Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}")
    pyplot.show()





def check_image(image, label, model):
    print(f'Label: {label}')
    prediction = model.predict(image.reshape(1, -1))
    print(f'Prediction {prediction}')
    image = image.reshape(200, 100, 3)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()