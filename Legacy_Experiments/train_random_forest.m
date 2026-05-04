% This script trains a Random Forest (Ensemble of Bagged Trees)
% on the original, highly successful FFT dataset.

clc;
clear all;
close all;

%% 1. Load the Data
fprintf('Loading original FFT dataset...\n');
load('dataset_fft_multiclass.mat');
y_labels = y_labels(:);

%% 2. Split into Training and Testing Sets
fprintf('Splitting data...\n');
cv = cvpartition(y_labels, 'HoldOut', 0.2);

X_train = X_features(training(cv), :);
y_train = y_labels(training(cv), :);

X_test = X_features(test(cv), :);
y_test = y_labels(test(cv), :);

%% 3. Train the Random Forest
fprintf('\nTraining the Random Forest model (this may take a moment)...\n');
% Random Forests (Bagged Trees) are incredibly good at ignoring noise in wide data
% They will find the exact frequency bins that identify a keystroke without getting 
% confused by the 1000 features.
t = templateTree('Reproducible', true);

% We use 150 "Trees" to vote on the best answer. 
% This is widely considered one of the best AI algorithms.
rf_model = fitcensemble(X_train, y_train, 'Method', 'Bag', 'NumLearningCycles', 150, 'Learners', t);

fprintf('Training complete!\n');

%% 4. Evaluate the Model
fprintf('\nEvaluating the model on the test set...\n');
y_pred = predict(rf_model, X_test);

accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('======================================\n');
fprintf('  TEST ACCURACY (Random Forest): %.2f%%\n', accuracy);
fprintf('======================================\n');

%% 5. Visualization: Confusion Matrix
figure('Name', 'RF Performance - Confusion Matrix');
confusionchart(y_test, y_pred);
title(sprintf('Random Forest Confusion Matrix (Accuracy: %.2f%%)', accuracy));

%% 6. Save Model
save('trained_rf_model.mat', 'rf_model');
