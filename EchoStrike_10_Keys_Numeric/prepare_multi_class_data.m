% =========================================================================
% EchoStrike: 10-Keys Numeric Dataset Preparation (0-9)
% =========================================================================
% Description:
% This script extracts features from 10 distinct keyboard keys (0 through 9).
% It applies the Fourier Transform, shifts to Logarithmic/dB scale, and 
% min-max normalizes the signal to generate the first 1000 uniform features.
% The output matrix is designed for the One-vs-All Multi-Class SVM.
% =========================================================================

clc; clear all; close all;

%% 1. Configuration & Open-Source Paths
target_length = 1000;  % Compress feature vector to exactly 1000 bins 

% TODO: Replace this path with the directory where your categorized audio is saved
base_path = 'path/to/your/audio_samples'; 

% 10 categories mapping to folders '0' through '9'
categories = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};

% Corresponding integer labels for One-vs-All SVM target mapping
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 

% Initialize the storage matrices
X_features = [];
y_labels = [];

%% 2. Audio Ingestion & Extraction Loop
for c = 1:length(categories)
    current_category = categories{c};
    current_label = labels(c);
    
    % Navigate to the specific class folder
    folder_path = fullfile(base_path, current_category);
    
    % Verify directory exists, otherwise skip
    if ~exist(folder_path, 'dir')
        warning('Folder %s does not exist. Skipping...', folder_path);
        continue;
    end
    
    % Get list of all .wav files in that folder
    file_list = dir(fullfile(folder_path, '*.wav'));
    fprintf('Processing folder: %s (%d files found)\n', current_category, length(file_list));
    
    for i = 1:length(file_list)
        % Read the physical audio file
        file_name = fullfile(folder_path, file_list(i).name);
        [x, Fe] = audioread(file_name);
        
        % Convert to mono channel
        x = mean(x, 2); 
        Te = 1/Fe;
        N_segment = length(x);
        
        %% 3. Fast Fourier Transform & The Golden Pipeline
        % Frequency vector computation
        f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
        X_segment = fftshift(fft(x) * Te);
        
        % Extract positive frequency spectrum
        pos_idx = find(f_segment >= 0);
        fft_magnitude = abs(X_segment(pos_idx));
        
        % Convert to Logarithmic Scale (dB)
        % eps acts as a safety boundary preventing log10(0) returning -Inf
        features_log = 20 * log10(fft_magnitude' + eps); 

        % Strict Min-Max Normalization [0, 1]
        features_log = features_log - min(features_log);
        features = features_log / (max(features_log) + eps);
        
        % Downsample explicitly down to 1000 points 
        features_resampled = resample(features, target_length, length(features));
        
        % Append into massive dataset
        X_features = [X_features; features_resampled];
        y_labels = [y_labels; current_label];
    end
end

%% 4. Final Summary & Output
fprintf('\n--- Extraction Complete ---\n');
fprintf('Total keystrokes processed: %d\n', size(X_features, 1));
fprintf('Feature vector size: %d\n', size(X_features, 2));
fprintf('Labels mapped: %d classes (0 to 9)\n', length(categories));

% Save the dataset for training the multi-class SVM
save('dataset_fft_10keys.mat', 'X_features', 'y_labels');
fprintf('Dataset successfully saved as dataset_fft_10keys.mat\n');

%% 5. Visualization: PCA Preview
% Render a plot showing a random sample profile for EACH class (0 through 9)
if ~isempty(X_features)
    figure('Name', 'Logarithmic FFT Features per Class', 'Position', [100, 100, 1200, 800]);
    
    for c = 1:length(categories)
        current_label = labels(c);
        
        % Locate sample index mathematically tied to current label
        idx_for_class = find(y_labels == current_label);
        
        if ~isempty(idx_for_class)
            % Pick one random sample representation from this class
            random_idx = idx_for_class(randi(length(idx_for_class)));
            
            % Plot via 2x5 grid matrix
            subplot(2, 5, c);
            plot(X_features(random_idx, :), 'LineWidth', 1);
            title(sprintf('Class %d', current_label));
            xlabel('Feature Index');
            ylabel('Log-Mag (Norm)');
            grid on;
        end
    end
end
