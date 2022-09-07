clear; close all; clc; clear all; close

%------LOAD DATASET
dataset = load('dataset.csv');

%------SHUFFLE DATASET
size_dataset = size(dataset);
m_dataset = size_dataset(1);
idx = randperm(m_dataset);
rand_dataset = dataset;
rand_dataset(idx, :) = dataset(:, :);
old_dataset = dataset;
dataset = rand_dataset;

%------SPLIT DATA INTO FEATURES AND TARGET
X_data = dataset(:, 1:4);
y_data = dataset(:, 5);

%------FEATURE NORMALIZATION
t = ones(length(X_data), 1);
X_norm = (X_data - (t * mean(X_data))) ./ (t * std(X_data));
y_log = log(1+y_data);


%------SPLIT DATA INTO TRAINING AND TEST SETS
X_train = X_norm(1:30, :);
y_train = y_data(1:30, :);
X_test = X_norm(31:42, :);
y_test = y_data(31:42, :);

X = X_train;
y = y_train;

MDL = fitrsvm(X,y,'epsilon',0.01,'kernelfunction','gaussian');

y_predict = predict(MDL, X_test);
%plot(y_predict, y_test, 'o'); hold on
%plot([0, 1], [0, 1], '-'); hold off

%--------RMSE
rmse = sqrt(mean((y_predict - y_test).^2))

%--------R-Squared
SSresid = sum((y_predict - y_test).^2);
SStotal = length(y_test)-1 * var(y_test);
R_sq = 1 - (SSresid/SStotal)

