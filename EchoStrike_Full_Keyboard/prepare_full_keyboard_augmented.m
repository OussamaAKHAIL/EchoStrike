% =========================================================================
% FULL KEYBOARD DATASET PREPARATION & DATA AUGMENTATION SCRIPT
% =========================================================================
% Project: Acoustic Side-Channel Attack Analysis on Keyboards
% 
% Description:
% This script automates the extraction and preparation of acoustic features 
% from mechanical keyboard sounds. It dynamically reads subfolders corresponding 
% to distinct key classes, processes the raw `.wav` clips using Fast Fourier 
% Transform (FFT), applies logarithmic scale normalization, and performs 
% Data Augmentation (White Noise & Pitch Shifting) to artificially triple 
% the size of the dataset. The resulting feature matrix is saved and ready 
% for AI classification models (like SVM or Random Forest).
% =========================================================================

clc; clear; close all;

%% 1. Configuration & Path Initialization
target_length = 1000; % Final number of frequency bins for our feature vector
base_path = 'path/to/your/segmented_audios';

% Read the directory to find all dynamically created subfolders
d = dir(base_path);
is_directory = [d.isdir];
folder_names = {d(is_directory).name};

% Filter out the standard hidden directories '.' and '..'
folder_names = folder_names(~ismember(folder_names, {'.', '..'}));

% Display the number of distinct key classes detected
fprintf('Found %d distinct key classes to process.\n', length(folder_names));

% Initialize the main arrays that will hold the dataset
X_features = []; % Matrix containing the spectral features (N x 1000)
y_labels = {};   % Cell array for categorical labels (e.g., "Rshift", "Space")

%% 2. Data Extraction & Augmentation Loop
% Iterate over every single key class folder
for c = 1:length(folder_names)
    current_category = folder_names{c};
    folder_path = fullfile(base_path, current_category);
    
    % Find all segmented .wav files in the current category folder
    file_list = dir(fullfile(folder_path, '*.wav'));
    fprintf('Processing [%s]... (Augmenting %d clips into %d clips)\n', ...
        current_category, length(file_list), length(file_list)*3);
    
    % Iterate over every audio clip in the folder
    for i = 1:length(file_list)
        file_name = fullfile(folder_path, file_list(i).name);
        
        % Read the audio file
        % x = amplitude values, Fe = sampling frequency (Hz)
        [x, Fe] = audioread(file_name);
        
        % Convert stereo to mono by averaging the two channels
        x_raw = mean(x, 2); 
        
        % -----------------------------------------------------------
        % DATA AUGMENTATION: Generate 3 versions of each audio file
        % This radically improves model robustness against new conditions
        % -----------------------------------------------------------
        % 1. Additive White Gaussian Noise (AWGN) - 1% amplitude variance
        x_noise = x_raw + (0.01 * randn(size(x_raw))); 
        
        % 2. Pitch Shifting - Emulates different typing forces/anomalies
        x_pitch = resample(x_raw, 98, 100); 
        
        % Store all three versions in a cell array for sequential processing
        audio_versions = {x_raw, x_noise, x_pitch};
        
        % Process each augmented version
        for v = 1:length(audio_versions)
            x_aug = audio_versions{v};
            
            %% 3. Spectral Processing 
            Te = 1/Fe; % Sampling period
            N_segment = length(x_aug); % Total samples in segment
            
            % Generate the frequency axis
            f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
            
            % Compute the Fast Fourier Transform (FFT) and center it
            X_segment = fftshift(fft(x_aug) * Te);
            
            % Extract only the positive half of the frequency spectrum
            pos_idx = find(f_segment >= 0);
            fft_magnitude = abs(X_segment(pos_idx));
            
            %% 4. The Golden Processing Pipeline
            % Convert the raw magnitude strictly to a logarithmic scale (dB)
            % The eps prevents log10(0) from returning -Inf
            features_log = 20 * log10(fft_magnitude' + eps);
            
            % Standard Min-Max Normalization
            % This forces all spectral shapes into a strict [0, 1] range, 
            % preventing highly energetic frequencies from dominating the model.
            features_log = features_log - min(features_log);
            features = features_log / (max(features_log) + eps);
            
            % Downsample the final feature array to exactly 1000 bins uniformly
            features_resampled = resample(features, target_length, length(features));
            
            % Append the final standardized feature vector into the main dataset
            X_features = [X_features; features_resampled];
            
            % Record the physical key name as the True Label for this feature
            y_labels = [y_labels; {current_category}];
        end
    end
end

%% 5. Final Dataset Packaging
% Ensure labels are formatted as a 'categorical' array for AI training compatibility
y_labels_categorical = categorical(y_labels);

fprintf('\n--- Processing Complete ---\n');
fprintf('Total keystrokes extracted across all keys (Augmented!): %d\n', size(X_features, 1));

% Save the massive feature matrix and labels for the training script
save('dataset_full_keyboard_augmented.mat', 'X_features', 'y_labels_categorical', '-v7.3');
fprintf('Saved successfully as dataset_full_keyboard_augmented.mat\n');
