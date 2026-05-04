% This script trains the SVM model on the AUGMENTED dataset.
% Because there is 3x more data, this training phase will take longer!

clc;
clear all;
close all;

%% 1. Load the Data
fprintf('Loading Augmented FFT dataset...\n');
load('dataset_augmented_multiclass.mat');
y_labels = y_labels(:);

%% 2. Split into Training and Testing Sets
fprintf('Splitting data...\n');
cv = cvpartition(y_labels, 'HoldOut', 0.2);

X_train = X_features(training(cv), :);
y_train = y_labels(training(cv), :);

X_test = X_features(test(cv), :);
y_test = y_labels(test(cv), :);

%% 3. Train the One-Vs-All SVM with Kernel
fprintf('\nTraining the SVM on Augmented data (Automated Optimization)...\n');
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);

% We use 30 evaluations again, but the learning itself will take longer 
% because there are triple the number of audio files to process.
opts = struct('MaxObjectiveEvaluations', 30, 'ShowPlots', true, 'Verbose', 1);

svm_model = fitcecoc(X_train, y_train, 'Learners', t, 'Coding', 'onevsall', ...
    'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts);

fprintf('Training complete!\n');

%% 4. Evaluate the Model
fprintf('\nEvaluating the model...\n');
y_pred = predict(svm_model, X_test);

accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('======================================\n');
fprintf('  TEST ACCURACY (Augmented): %.2f%%\n', accuracy);
fprintf('======================================\n');

%% 5. Visualization: Confusion Matrix
figure('Name', 'SVM Performance (Augmented) - Confusion Matrix');
confusionchart(y_test, y_pred);
title(sprintf('Augmented + One-Vs-All Kernel SVM (Accuracy: %.2f%%)', accuracy));

%% 6. Save the Trained Model
save('trained_svm_model_augmented.mat', 'svm_model');
fprintf('\nModel saved as trained_svm_model_augmented.mat\n');
