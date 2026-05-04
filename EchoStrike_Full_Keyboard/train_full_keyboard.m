% ========================================================
% FULL KEYBOARD SVM TRAINING
% ========================================================
% Trains the Hyperparameter Optimized One-vs-All SVM on the full 
% dynamically extracted dataset.

clc; clear; close all;

%% 1. Load the Data
fprintf('Loading Full Keyboard Augmented Dataset...\n');
load('dataset_full_keyboard_augmented.mat');

%% 2. Split into Training and Testing Sets
fprintf('Splitting data...\n');
% Because y_labels_categorical is a literal text category array, cvpartition natively balances it!
cv = cvpartition(y_labels_categorical, 'HoldOut', 0.2);

X_train = X_features(training(cv), :);
y_train = y_labels_categorical(training(cv), :);

X_test = X_features(test(cv), :);
y_test = y_labels_categorical(test(cv), :);

%% 3. Train the One-Vs-All SVM (Single Pass)
fprintf('\nTraining the SVM on Full Dataset (This will only run ONE time)...\n');

% The parameters from yesterday were optimized for 10 classes! 
% Applying them to 104 classes squeezed the math too tight, collapsing accuracy to 1%.
% We will let MATLAB heuristically calculate the best 'auto' scale for the full 104 classes.
t = templateSVM('KernelFunction', 'gaussian', ...
                'KernelScale', 'auto', ...
                'Standardize', true);

% Train directly without 'OptimizeHyperparameters' (Trains only 1 cycle)
svm_model = fitcecoc(X_train, y_train, 'Learners', t, 'Coding', 'onevsall');

fprintf('Training complete!\n');

%% 4. Evaluate the Model
fprintf('\nEvaluating the model...\n');
y_pred = predict(svm_model, X_test);

accuracy = sum(y_pred == y_test) / length(y_test) * 100;
fprintf('======================================\n');
fprintf('  TEST ACCURACY (All Keys): %.2f%%\n', accuracy);
fprintf('======================================\n');

%% 5. Visualization: Confusion Matrix
figure('Name', 'SVM Full Keyboard - Confusion Matrix');
confusionchart(y_test, y_pred);
title(sprintf('Full Keyboard Kernel SVM (Accuracy: %.2f%%)', accuracy));

%% 6. Save Model
% The full keyboard math model is massive! We must specify -v7.3 to save files over 2GB.
save('trained_svm_full_keyboard.mat', 'svm_model', '-v7.3');
fprintf('\nModel saved successfully as trained_svm_full_keyboard.mat\n');

%% 7. Visualize Feature Separation in 3D Space (PCA)
% To visualize 1000-dimensional audio features, we compress them down to 3D using PCA.
% This creates a 3D scatter plot so you can see exactly how the keys form clusters!
fprintf('\nGenerating 3D Visualization of the Feature Space...\n');

% Apply Principal Component Analysis (Reduce 1000D -> 3D)
[~, score, ~] = pca(X_features);

% Extract the first 3 principal components
X_3D = score(:, 1:3);

% Convert string/categorical labels to raw numbers for color mapping
numeric_labels = double(y_labels_categorical); 

% Plot the 3D Scatter
figure('Name', '3D Acoustic Feature Space', 'Position', [200, 200, 800, 600]);
scatter3(X_3D(:, 1), X_3D(:, 2), X_3D(:, 3), 15, numeric_labels, 'filled', 'MarkerFaceAlpha', 0.6);

% Make it beautifully readable
colormap(turbo); % 'turbo' gives excellent distinct colors for 100+ classes
title('3D Visualization of Keyboard Acoustic Features (PCA)');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
zlabel('Principal Component 3');
grid on;

% Set a good camera angle to view the data spread
view(45, 30);
fprintf('3D Visualization successfully rendered!\n');
