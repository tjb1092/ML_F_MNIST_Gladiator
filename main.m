clc;clear all; close all; % Clean-up
addpath(genpath('drtoolbox')); %Add Data Visualization Tools.

%% Import Data
[x_train, image_train] = loadMNISTImages('train-images-idx3-ubyte');
x_train = x_train';
y_train = loadMNISTLabels('train-labels-idx1-ubyte');

[x_test, image_test] = loadMNISTImages('t10k-images-idx3-ubyte');
x_test = x_test';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');


%% Visualize the Data.
visualizeData(x_test,y_test, image_test);

%% Perform Multi-Class SVM Classification

SVM_accuracy = ECOC_Classifier(x_train, y_train, x_test, y_test);
%% Perform K - Nearest Neighbors Classification
%Initial grid-search to find best distance metric
[KNN_Accuracy, KNNModel, KNN_GridSearch ] = KNN_Classifier( x_train, y_train, x_test, y_test, 0);
%Second grid-search to find the best k.

%% Perform Naive Bayes Classification

NB_accuracy = NB_Classifier(x_train, y_train, x_test, y_test);
%% Perform Multi-Class Logistic Regression



