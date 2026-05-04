% ========================================================
% FULL KEYBOARD DATASET PREPARATION
% ========================================================
% This script automatically reads ANY folder name inside your base path,
% meaning it scales seamlessly to 10 keys or all 104 keys.
% It uses your champion logic: FFT -> Logarithmic Scale -> Normalization.

clc; clear; close all;

%% Configuration
target_length = 1000;
% TODO: Replace with the path to your dynamically segmented audio folders
base_path = 'path/to/your/segmented_audios';

% Retrieve all dynamically created subfolders
d = dir(base_path);
is_directory = [d.isdir];
folder_names = {d(is_directory).name};
folder_names = folder_names(~ismember(folder_names, {'.', '..'}));

fprintf('Found %d distinct key classes to process.\n', length(folder_names));

X_features = [];
y_labels = {}; % We use a cell array for labels so it can naturally store text like "Rshift" or "Enter"

%% Data Extraction Loop
for c = 1:length(folder_names)
    current_category = folder_names{c};
    folder_path = fullfile(base_path, current_category);
    
    file_list = dir(fullfile(folder_path, '*.wav'));
    fprintf('Processing [%s]... (%d clips)\n', current_category, length(file_list));
    
    for i = 1:length(file_list)
        file_name = fullfile(folder_path, file_list(i).name);
        [x, Fe] = audioread(file_name);
        
        x = mean(x, 2); % Convert to Mono
        Te = 1/Fe;
        N_segment = length(x);
        f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
        X_segment = fftshift(fft(x) * Te);
        
        pos_idx = find(f_segment >= 0);
        fft_magnitude = abs(X_segment(pos_idx));
        
        % The Golden Pipeline: Log scale -> Min Subtraction -> Max Division
        features_log = 20 * log10(fft_magnitude' + eps);
        features_log = features_log - min(features_log);
        features = features_log / (max(features_log) + eps);
        
        % Downsample intelligently to exactly 1000 bins
        features_resampled = resample(features, target_length, length(features));
        
        X_features = [X_features; features_resampled];
        
        % Append the text name directly as the label!
        y_labels = [y_labels; {current_category}];
    end
end

%% Clean up and Save
% Convert strings into a 'categorical' array. MATLAB's fitcecoc SVM loves this format.
y_labels_categorical = categorical(y_labels);

fprintf('\n--- Processing Complete ---\n');
fprintf('Total keystrokes extracted across all keys: %d\n', size(X_features, 1));

save('dataset_full_keyboard.mat', 'X_features', 'y_labels_categorical');
fprintf('Saved successfully as dataset_full_keyboard.mat\n');
