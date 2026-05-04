% =========================================================================
% EchoStrike: Binary Support Vector Machine (SVM) Training + PCA
% =========================================================================
% Description:
% This script imports the binary (Space vs Shift) dataset and shrinks the 
% 1000-point FFT array using Principal Component Analysis (PCA). 
% This strips away acoustic background noise before feeding the core signals
% into a multi-class SVM structure (used as a testing framework).
% =========================================================================

clc; clear all; close all;

%% 1. Load the Binary Dataset
fprintf('Loading binary dataset...\n');
% Loads the output from prepare_binary_data.m
load('dataset_binary_fft.mat'); 
y_labels = y_labels(:); % Ensure it's a strict column vector

%% 2. Split into Training and Testing Sets
fprintf('Splitting data into 80%% training and 20%% testing...\n');
cv = cvpartition(y_labels, 'HoldOut', 0.2);

X_train = X_features(training(cv), :);
y_train = y_labels(training(cv), :);

X_test = X_features(test(cv), :);
y_test = y_labels(test(cv), :);

fprintf('Training samples: %d\n', sum(training(cv)));
fprintf('Testing samples: %d\n', sum(test(cv)));

%% 3. Apply PCA (Principal Component Analysis)
fprintf('\n--- Applying PCA Dimensionality Reduction ---\n');
% Calculate PCA on training data only to prevent data leakage/bias
[coeff, score, ~, ~, explained, mu] = pca(X_train);

% Keep enough mathematical components to explain 95% of the data's true variance
cumulative_variance = cumsum(explained);
num_components = find(cumulative_variance >= 95, 1);
fprintf('Reduced from %d features down to just %d features (maintaining 95%% of variance)\n', ...
    size(X_train, 2), num_components);

% Project training data down to the new smaller size
X_train_pca = score(:, 1:num_components);

% Project test data using the exact same mathematical transformation
X_test_centered = X_test - mu;
X_test_pca = X_test_centered * coeff(:, 1:num_components);

%% 4. Train the SVM with Gaussian Kernel
fprintf('\nTraining the SVM model on PCA reduced data...\n');
% Set up the SVM template (RBF Kernel)
t = templateSVM('KernelFunction', 'gaussian', 'KernelScale', 'auto', 'Standardize', true);

% Set optimization options (limit to 30 evaluations for speed)
opts = struct('MaxObjectiveEvaluations', 30, 'ShowPlots', true, 'Verbose', 1);

% Train the model using fitcecoc ON THE PCA DATA
svm_model = fitcecoc(X_train_pca, y_train, 'Learners', t, 'Coding', 'onevsall', ...
    'OptimizeHyperparameters', 'auto', 'HyperparameterOptimizationOptions', opts);

fprintf('Training complete!\n');

%% 5. Evaluate the Model
fprintf('\nEvaluating the PCA model on the test set...\n');
% We MUST pass the reduced 'X_test_pca' into the predict function!
y_pred = predict(svm_model, X_test_pca);

% Calculate accuracy against the test subset
accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('======================================\n');
fprintf('  TEST ACCURACY (with PCA): %.2f%%\n', accuracy);
fprintf('======================================\n');

%% 6. Visualization: Confusion Matrix
figure('Name', 'SVM Performance (Binary PCA) - Confusion Matrix');
confusionchart(y_test, y_pred);
title(sprintf('PCA + SVM Binary (Accuracy: %.2f%%)', accuracy));

%% 7. Save Model & PCA Coefficients
% We must save the PCA coefficients along with the model so any future predictions 
% on brand new audio files can be shrunk the exact same way before passing to the SVM!
save('trained_svm_binary_pca.mat', 'svm_model', 'coeff', 'mu', 'num_components');
fprintf('\nModel and PCA parameters saved as trained_svm_binary_pca.mat\n');
