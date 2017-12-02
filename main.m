clc;clear all; close all; % Clean-up
addpath(genpath('drtoolbox')); %Add Data Visualization Tools.

%% Import Data
[x_train, image_train] = loadMNISTImages('train-images-idx3-ubyte');
x_train = x_train';
y_train = loadMNISTLabels('train-labels-idx1-ubyte');

[x_test, image_test] = loadMNISTImages('t10k-images-idx3-ubyte');
x_test = x_test';
y_test = loadMNISTLabels('t10k-labels-idx1-ubyte');

%% Standardizing the pixel data
x_test = x_test / 255.0 * 2 - 1;
x_train = x_train / 255.0 * 2 - 1;

%% Visualize the Data.
visualizeData(x_test,y_test, image_test);

%% Perform Multi-Class SVM Classification

SVM_accuracy = ECOC_Classifier(x_train, y_train, x_test, y_test);
%% Perform K - Nearest Neighbors Classification

%% Perform Naive Bayes Classification

%% Perform Multi-Class Logistic Regression



