% =========================================================================
% EchoStrike: Binary Basic Data Preparation (Space vs Shift)
% =========================================================================
% This is the foundational script proving acoustic keystroke extraction.
% It isolates the Fast Fourier Transform (FFT) features for exactly two 
% keys (Right Shift and Space) and builds the baseline .mat dataset.
% =========================================================================

clc; clear all; close all;

%% 1. Configuration & Open-Source Paths
target_length = 1000;  % Compress feature vector to exactly 1000 bins 

% TODO: Replace this path with the directory where your categorized audio is saved
base_path = 'path/to/your/audio_samples'; 

% Define the two fundamental classes we are testing
categories = {'right_shift', 'space'};
labels = [0, 1]; % Mathematical targets: 0 = Right Shift, 1 = Space

% Initialize the storage matrices
X_features = [];
y_labels = [];

%% 2. Audio Ingestion & Extraction Loop
for c = 1:length(categories)
    current_category = categories{c};
    current_label = labels(c);
    
    % Navigate to the specific class folder
    folder_path = fullfile(base_path, current_category);
    file_list = dir(fullfile(folder_path, '*.wav'));
    
    fprintf('Processing folder: %s (%d files found)\n', current_category, length(file_list));
    
    for i = 1:length(file_list)
        % Read the physical audio file
        file_name = fullfile(folder_path, file_list(i).name);
        [x, Fe] = audioread(file_name);
        
        % Convert to mono channel to standardize the wave
        x = mean(x, 2); 
        Te = 1/Fe; % Sampling period
        N_segment = length(x);
        
        %% 3. Fast Fourier Transform (FFT)
        % Generate frequency axis
        f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
        X_segment = fftshift(fft(x) * Te);
        
        % Filter out negative frequencies (mathematical mirror)
        pos_idx = find(f_segment >= 0);
        fft_magnitude = abs(X_segment(pos_idx));
        
        % Normalize magnitude so high volumes don't break the SVM
        % (eps is a safe switch to avoid dividing by absolute zero)
        features = fft_magnitude' / (max(fft_magnitude) + eps);
        
        % Resample explicitly down to 1000 points to keep matrix dimensions uniform
        features_resampled = resample(features, target_length, length(features));
        
        % Append the row into the main dataset
        X_features = [X_features; features_resampled];
        y_labels = [y_labels; current_label];
    end
end

%% 4. Final Output Packaging
fprintf('\n--- Extraction Complete ---\n');
fprintf('Total keystrokes processed: %d\n', size(X_features, 1));
fprintf('Feature vector size: %d\n', size(X_features, 2));

% Save the baseline dataset for the Binary SVM training script
save('dataset_binary_fft.mat', 'X_features', 'y_labels');
fprintf('Dataset successfully saved as dataset_binary_fft.mat\n');
