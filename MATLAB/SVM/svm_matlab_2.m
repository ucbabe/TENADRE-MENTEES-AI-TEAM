close all; clc;

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

%------SPLIT DATA INTO TRAINING AND TEST SETS
data_train = dataset(1:30, :);
data_test = dataset(31:42, :);

%---------MODEL
[trainedModel, validationRMSE] = svm_m1(data_test);
y_predict = trainedModel.predictFcn(data_test(:, 1:4));

y_test = data_test(:, 5);

%plot(y_predict, y_test, 'o'); hold on
%plot([0, 1], [0, 1], '-'); hold off

%--------RMSE
rmse = sqrt(mean((y_predict - y_test).^2))

%--------R-Squared
SSresid = sum((y_predict - y_test).^2);
SStotal = length(y_test)-1 * var(y_test);
R_sq = 1 - (SSresid/SStotal)