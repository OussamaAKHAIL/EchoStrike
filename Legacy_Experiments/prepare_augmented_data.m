% DATA AUGMENTATION (The final step for high accuracy)
% This script takes your existing audio files and automatically generates 
% slightly modified "fake" copies of them (adding noise and pitch shifting).
% This gives the SVM 3x more data to learn from, making it much smarter!

clc;
clear all;
close all;

%% Configuration
target_length = 1000;
base_path = 'C:\Users\akous\Downloads\MKA datasets\segmented audios'; 
categories = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 

X_features = [];
y_labels = [];

%% Loop through each category folder
for c = 1:length(categories)
    current_category = categories{c};
    current_label = labels(c);
    
    folder_path = fullfile(base_path, current_category);
    file_list = dir(fullfile(folder_path, '*.wav'));
    
    fprintf('Processing folder %s (Augmenting %d files into %d files)...\n', ...
        current_category, length(file_list), length(file_list)*3);
    
    for i = 1:length(file_list)
        file_name = fullfile(folder_path, file_list(i).name);
        [x_raw, Fe] = audioread(file_name);
        x_raw = mean(x_raw, 2); % Convert to mono
        
        % Define 3 versions of the audio to extract:
        % 1. The original pristine audio
        % 2. Audio with background static (White Noise)
        % 3. Audio very slightly pitched down
        
        x_noise = x_raw + (0.01 * randn(size(x_raw))); % Add 1% white noise
        x_pitch = resample(x_raw, 98, 100); % Pitch shift slightly
        
        audio_versions = {x_raw, x_noise, x_pitch};
        
        for v = 1:length(audio_versions)
            x = audio_versions{v};
            
            Te = 1/Fe;
            N_segment = length(x);
            f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
            X_segment = fftshift(fft(x) * Te);
            
            pos_idx = find(f_segment >= 0);
            fft_magnitude = abs(X_segment(pos_idx));
            
            % Logarithmic Scale & Normalize
            features_log = 20 * log10(fft_magnitude' + eps); 
            features_log = features_log - min(features_log);
            features = features_log / (max(features_log) + eps);
            
            % Resample to 1000 features
            features_resampled = resample(features, target_length, length(features));
            
            % Add to dataset
            X_features = [X_features; features_resampled];
            y_labels = [y_labels; current_label];
        end
    end
end

fprintf('\n--- Augmentation Complete ---\n');
fprintf('Total samples grew to: %d\n', size(X_features, 1));
save('dataset_augmented_multiclass.mat', 'X_features', 'y_labels');
