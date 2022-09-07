close all; clc; clear;

addpath('codes');

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

%--------SPLIT DATA INTO FEATURES AND TARGET
X_data = dataset(:, 1:4);
y_data = dataset(:, 5);

%--------FEATURE NORMALIZATION
t = ones(length(X_data), 1);
X_norm = (X_data - (t * mean(X_data))) ./ (t * std(X_data));
y_log = log(1+y_data);
X = X_norm;
y = y_data;

%--------SPLIT DATA INTO TRAINING AND TEST SETS
%train_data = data(1:30, :);
%test_data = data(31:42, :);

%[TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy] = ELM(train_data, test_data, 0, 15, 'sig')


%-----------------------
X_train = X_norm(1:30, :);
y_train = y_data(1:30, :);
X_test = X_norm(31:42, :);
y_test = y_data(31:42, :);
%-----------------------

% define Options
Opts.ELM_Type='Regrs';    % 'Class' for classification and 'Regrs' for regression
Opts.number_neurons=15;  % Maximam number of neurons 
Opts.Tr_ratio=1.00;       % training ratio
Opts.Bn=0;                % 1 to encode  lables into binary representations
                          % if it is necessary

% Training
[net]= elm_LB(X_train, y_train,Opts);

% prediction
output=elmPredict(net,X_test);

y_predict = output;

%plot(output, y_test, 'o'); hold on
%plot([0 1], [0 1], '-'); hold off


%--------RMSE
rmse = sqrt(mean((y_predict - y_test).^2))

%--------R-Squared
SSresid = sum((y_predict - y_test).^2);
SStotal = length(y_test)-1 * var(y_test);
R_sq = 1 - (SSresid/SStotal)