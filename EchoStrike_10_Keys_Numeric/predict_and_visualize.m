% =========================================================================
% EchoStrike: 10-Keys Real-time Prediction & Heatmap Visualization
% =========================================================================
% Description:
% This script simulates a real-time side-channel attack on the Numpad (0-9). 
% It selects a random audio file, processes the FFT signature, and queries 
% the trained SVM. It then dynamically generates an interactive 2D 
% Topographic Heatmap to visually map the AI's probability distribution 
% physically across the keys.
% =========================================================================

clc; clear all; close all;

%% 1. Load the trained SVM model
fprintf('Loading Trained SVM Model...\n');
% Load the 10-keys model generated from train_multiclass_svm.m
load('trained_svm_10keys.mat');

%% 2. Select a test audio file
fprintf('Grabbing a random audio file for prediction...\n');
% TODO: Replace this path with your root audio directory
base_path = 'path/to/your/audio_samples'; 
categories = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

% Safely pick a random category that actually has .wav files inside
file_list = [];
while isempty(file_list)
    truth_class = categories{randi(10)};
    folder_path = fullfile(base_path, truth_class);
    
    if exist(folder_path, 'dir')
        file_list = dir(fullfile(folder_path, '*.wav'));
    else
        file_list = [];
    end
end
test_file = fullfile(folder_path, file_list(randi(length(file_list))).name);

fprintf('--> Actually pressed key: %s\n', truth_class);

%% 3. Process the audio mathematically (Exact same as preparation phase)
[x, Fe] = audioread(test_file);
x = mean(x, 2); % Convert to Mono
Te = 1/Fe;
N_segment = length(x);
f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
X_segment = fftshift(fft(x) * Te);

pos_idx = find(f_segment >= 0);
fft_magnitude = abs(X_segment(pos_idx));

features_log = 20 * log10(fft_magnitude' + eps);
features_log = features_log - min(features_log);
features = features_log / (max(features_log) + eps);
features_resampled = resample(features, 1000, length(features));

%% 4. Have the SVM Make a Prediction
% The model returns the predicted label AND a raw mathematical "score" for each class
[pred_label, scores] = predict(svm_model, features_resampled);

% Convert the raw SVM scores into clean Percentages (0% to 100%)
% We use "Softmax" probability math natively
temperature = 0.5; % Hyperparameter to smooth out probability distributions
s = scores / temperature;
s = s - max(s);
probs = exp(s) ./ sum(exp(s));
probs = probs * 100; % Convert to standard percentage

fprintf('--> AI Predicted Key: %d\n', pred_label);

%% 5. Visualize the Continuous Heatmap
% Let's draw a perfectly connected, smooth heat surface across the entire physical board.

% Create a nice clean window
f = figure('Name', 'Keyboard AI Prediction Matrix', 'Position', [100, 100, 450, 600], 'Color', [1 1 1]);

% Create a title displaying the result
title_txt = sprintf('AI Predicted: [%d]  |  Actual Key: [%s]', pred_label, truth_class);
annotation('textbox', [0 0.88 1 0.1], 'String', title_txt, 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', 'black', 'FontWeight', 'bold');

% Create axes for the grid
ax = axes('Position', [0.15 0.1 0.7 0.75]);
hold on; axis off; axis equal; 
xlim([0.5, 3.5]); ylim([0.5, 4.5]);

% --- 5a. Define the Math for the Heatmap ---
% We create a mathematical 2D matrix (Z) representing the probability at each physical key
get_prob = @(k) probs(svm_model.ClassNames == k);

Z = zeros(4, 3);
Z(4, 1) = get_prob(7); Z(4, 2) = get_prob(8); Z(4, 3) = get_prob(9);
Z(3, 1) = get_prob(4); Z(3, 2) = get_prob(5); Z(3, 3) = get_prob(6);
Z(2, 1) = get_prob(1); Z(2, 2) = get_prob(2); Z(2, 3) = get_prob(3);

% For the bottom row, we just spread the probability of "0" completely across it
p0 = get_prob(0);
Z(1, 1) = p0; Z(1, 2) = p0; Z(1, 3) = p0; 

% --- 5b. Generate a Topographic Map (Spline Interpolation) ---
% We use a high-resolution grid (200x200) to make the heat cloud blend smoothly
[X, Y] = meshgrid(1:3, 1:4);
[Xq, Yq] = meshgrid(linspace(0.5, 3.5, 200), linspace(0.5, 4.5, 200));

% Spline interpolation connects the discrete probability points continuously
Zq = interp2(X, Y, Z, Xq, Yq, 'spline');
Zq(Zq < 0) = 0; % Prevent math overshoot
Zq(Zq > 100) = 100;

% Draw the Topographic Heatmap
% '15' levels creates distinct, wavy "mountain" topographic lines
contourf(Xq, Yq, Zq, 15, 'LineColor', [0 0.3 0], 'LineWidth', 1.2); 

% Custom Green Colormap (White -> Light Green -> Dark Green)
c_levels = 256;
custom_green = [linspace(0.95, 0, c_levels)', linspace(1, 0.4, c_levels)', linspace(0.95, 0, c_levels)'];
colormap(custom_green);

% --- 5c. Draw the physical 3x4 Grid over the Heatmap ---
% Draw Vertical lines
for x = 0.5:1:3.5
    plot([x x], [0.5 4.5], 'w-', 'LineWidth', 3); 
end
% Draw Horizontal lines
for y = 0.5:1:4.5
    plot([0.5 3.5], [y y], 'w-', 'LineWidth', 3);
end

% --- 5d. Label formatting (Numbers & Percentages) ---
% Top 3 Rows
keys_to_draw = [7, 8, 9; 4, 5, 6; 1, 2, 3];
for row = 1:3
    for col = 1:3
        k = keys_to_draw(row, col);
        y_loc = 5 - row; % Map row 1 -> y=4, row 2 -> y=3, etc.
        
        text(col, y_loc + 0.1, num2str(k), 'FontSize', 24, 'FontWeight', 'bold', ...
            'Color', 'white', 'HorizontalAlignment', 'center');
        text(col, y_loc - 0.15, sprintf('%.1f%%', get_prob(k)), 'FontSize', 12, ...
            'Color', 'white', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
    end
end

% Bottom Row (Key 0 in the middle)
text(2, 1 + 0.1, '0', 'FontSize', 24, 'FontWeight', 'bold', ...
    'Color', 'white', 'HorizontalAlignment', 'center');
text(2, 1 - 0.15, sprintf('%.1f%%', p0), 'FontSize', 12, ...
    'Color', 'white', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');

hold off;
