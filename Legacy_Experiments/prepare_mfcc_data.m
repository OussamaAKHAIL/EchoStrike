% This code extracts MFCC (Mel-Frequency Cepstral Coefficients) 
% instead of raw FFT. MFCCs are designed to mimic human hearing and 
% usually result in much higher accuracy for audio classification!

clc;
clear all;
close all;

%% Configuration
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
    
    fprintf('Processing folder %s (%d files)...\n', current_category, length(file_list));
    
    for i = 1:length(file_list)
        % Read the audio file
        file_name = fullfile(folder_path, file_list(i).name);
        [x, Fe] = audioread(file_name);
        
        % Pre-processing: Convert to mono
        x = mean(x, 2); 
        
        % =======================================================
        % NEW FEATURE EXTRACTION: MFCC instead of Raw FFT 
        % =======================================================
        % MFCCs split the audio into windows and extract the most 
        % mathematically important acoustic frequencies.
        try
            coeffs = mfcc(x, Fe);
            
            % 'coeffs' is a matrix of windows over time.
            % We will squash it down to a single 1D array by taking 
            % the average (mean) and standard deviation of those windows.
            mean_coeffs = mean(coeffs, 1);
            std_coeffs = std(coeffs, 0, 1);
            
            % Combine them to create our final feature vector
            % This will give us a very clean, noiseless ~28 feature array
            % instead of 1000 noisy FFT bins!
            features = [mean_coeffs, std_coeffs];
            
            % Add to dataset
            X_features = [X_features; features];
            y_labels = [y_labels; current_label];
        catch
            fprintf('Error extracting MFCC for %s. Skipping...\n', file_name);
        end
    end
end

%% Final Summary
fprintf('\n--- MFCC Extraction Complete ---\n');
fprintf('Total samples: %d\n', size(X_features, 1));
fprintf('Feature vector length: %d\n', size(X_features, 2));

% Save the dataset for the training script
save('dataset_mfcc_multiclass.mat', 'X_features', 'y_labels');
