close all; clc; clear;

%--------LOAD DATASET
dataset = load('dataset.csv');

%--------SHUFFLE ROWS IN DATASET
size_dataset = size(dataset); 
m_dataset = size_dataset(1);
idx = randperm(m_dataset);
rand_dataset = dataset;
rand_dataset(idx, :) = dataset(:, :);
old_dataset = dataset;
dataset = rand_dataset;

%--------SPLIT DATA INTO TRAINING AND TEST SETS
train_data = dataset(1:30, :);
test_data = dataset(31:42, :);


X_train = train_data(:, 1:4);
y_train = train_data(:, 5);
X_test = test_data(:, 1:4);
y_test = test_data(:, 5);

%---------TRAIN MODEL
% anfisedit

anfis_mdl = readfis('anfis_mdl.fis');

y_predict = evalfis(anfis_mdl, X_test);

%--------RMSE
rmse = sqrt(mean((y_predict - y_test).^2))

%--------R-Squared
SSresid = sum((y_predict - y_test).^2);
SStotal = length(y_test)-1 * var(y_test);
R_sq = 1 - (SSresid/SStotal)

plot(y_predict, y_test, 'o'); hold on
plot([0 1], [0 1], '-'); hold off