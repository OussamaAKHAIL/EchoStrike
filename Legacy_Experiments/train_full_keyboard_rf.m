% ========================================================
% FULL KEYBOARD RANDOM FOREST TRAINING 
% ========================================================
% SVMs are incredibly powerful for 10-classes, but struggle 
% massively with 104 classes and 30,000+ clips. 
% This uses an Ensembled Decision Tree (Random Forest) which 
% trains exponentially faster (minutes instead of hours).

clc; clear; close all;

%% 1. Load Data
fprintf('Loading Full Keyboard Augmented Dataset...\n');
load('dataset_full_keyboard_augmented.mat');

%% 2. Split Data
fprintf('Splitting data...\n');
cv = cvpartition(y_labels_categorical, 'HoldOut', 0.2);

X_train = X_features(training(cv), :);
y_train = y_labels_categorical(training(cv), :);

X_test = X_features(test(cv), :);
y_test = y_labels_categorical(test(cv), :);

%% 3. Train Random Forest (TreeBagger)
fprintf('\nTraining the Random Forest (200 Trees) on Full Dataset...\n');
fprintf('This scales beautifully and should take exponentially less time than SVM!\n');

% Train 200 interconnected decision trees
numTrees = 200; 
rf_model = TreeBagger(numTrees, X_train, y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on');

fprintf('Training complete!\n');

%% 4. Evaluate the Model
fprintf('\nEvaluating the model...\n');
y_pred_cell = predict(rf_model, X_test);

% Convert predictions from cell array of chars back to categorical array
y_pred = categorical(y_pred_cell);

accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('======================================\n');
fprintf('  TEST ACCURACY (All Keys): %.2f%%\n', accuracy);
fprintf('======================================\n');

%% 5. Visualization: Confusion Matrix
figure('Name', 'Random Forest Full Keyboard - Confusion Matrix');
confusionchart(y_test, y_pred);
title(sprintf('Full Keyboard Random Forest (Accuracy: %.2f%%)', accuracy));

%% 6. Save Model
% Saving massive RF models correctly
save('trained_rf_full_keyboard.mat', 'rf_model', '-v7.3');
fprintf('\nModel saved successfully as trained_rf_full_keyboard.mat\n');
