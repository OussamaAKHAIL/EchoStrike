% This script trains a One-Vs-All Multi-Class SVM using an RBF (Gaussian) Kernel.
% It uses the dataset prepared and saved by prepare_multi_class_data.m.

clc;
clear all;
close all;

%% 1. Load the Data
fprintf('Loading dataset...\n');
% Load X_features and y_labels
load('dataset_fft_multiclass.mat');

% Ensure y_labels is a column vector (required for classification)
y_labels = y_labels(:);

%% 2. Split into Training and Testing Sets (80% Train, 20% Test)
fprintf('Splitting data into 80%% training and 20%% testing...\n');
cv = cvpartition(y_labels, 'HoldOut', 0.2);

X_train = X_features(training(cv), :);
y_train = y_labels(training(cv), :);

X_test = X_features(test(cv), :);
y_test = y_labels(test(cv), :);

fprintf('Training samples: %d\n', sum(training(cv)));
fprintf('Testing samples: %d\n', sum(test(cv)));

%% 3. Train the One-Vs-All SVM with Kernel
fprintf('\nTraining the One-vs-All SVM model (this may take a moment)...\n');

% Set up the SVM template with a Kernel (Gaussian/RBF is standard) 
% 'KernelScale', 'auto' is our baseline, but we will optimize around it
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);

% Set optimization options (limit to 30 evaluations so it doesn't take forever)
opts = struct('MaxObjectiveEvaluations', 30, 'ShowPlots', true, 'Verbose', 1);

% Train the multiclass model using fitcecoc
% 'OptimizeHyperparameters', 'auto' makes MATLAB automatically search for the 
% mathematical settings that result in the highest accuracy!
svm_model = fitcecoc(X_train, y_train, 'Learners', t, 'Coding', 'onevsall', ...
    'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts);

fprintf('Training complete!\n');

%% 4. Evaluate the Model
fprintf('\nEvaluating the model on the test set...\n');
y_pred = predict(svm_model, X_test);

% Calculate accuracy
accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('======================================\n');
fprintf('        TEST ACCURACY: %.2f%%\n', accuracy);
fprintf('======================================\n');

%% 5. Visualization: Confusion Matrix
% Display a confusion matrix to show exactly which keys were predicted correctly
figure('Name', 'SVM Performance - Confusion Matrix');
confusionchart(y_test, y_pred);
title(sprintf('One-Vs-All Kernel SVM Confusion Matrix (Accuracy: %.2f%%)', accuracy));

%% 6. Save the Trained Model
% Save the model so it can be used later without retraining
save('trained_svm_model.mat', 'svm_model');
fprintf('\nModel saved as trained_svm_model.mat\n');
