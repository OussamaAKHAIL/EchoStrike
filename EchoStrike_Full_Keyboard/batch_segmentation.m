% =========================================================================
% EchoStrike: Batch Automatic Audio Segmentation
% =========================================================================
% Description:
% This script automatically loops through raw audio recordings of typing 
% sessions. It uses dataset-specific limits (e.g., threshold voltage, 
% duration) to detect keystroke peaks, slice the audio around the peak, 
% and dump the standardized clips into specific categorized folders.
% =========================================================================

clc; clear; close all;

%% 1. Configuration & Open-Source Paths
% TODO: Replace these with the paths to where you store your raw audio data
output_base_folder = 'path/to/your/segmented_audios_output';

% Define datasets and their unique parameters
datasets = struct();

% --- HP Keyboard Parameters ---
datasets(1).name = 'HP';
datasets(1).path = 'path/to/your/HP/Raw_Data';
datasets(1).threshold = 0.055;
datasets(1).min_dist_seconds = 0.3;
datasets(1).segment_duration = 0.44;
datasets(1).pre_trigger = 0.1;

% --- Lenovo Keyboard Parameters ---
datasets(2).name = 'Lenovo';
datasets(2).path = 'path/to/your/Lenovo/Raw_Data';
datasets(2).threshold = 0.020;
datasets(2).min_dist_seconds = 0.3;
datasets(2).segment_duration = 0.3;
datasets(2).pre_trigger = 0.1;


%% 2. Process Each Dataset
for d = 1:length(datasets)
    ds = datasets(d);
    fprintf('\n======================================\n');
    fprintf('Processing %s Keyboard Dataset...\n', ds.name);
    fprintf('======================================\n');
    
    % Check if the Raw Data path exists
    if ~exist(ds.path, 'dir')
        fprintf('WARNING: Cannot find raw data folder: %s\n', ds.path);
        continue;
    end
    
    % Get a list of all .wav files in the raw data folder
    wav_files = dir(fullfile(ds.path, '*.wav'));
    
    if isempty(wav_files)
        fprintf('No .wav files found in this folder.\n');
        continue;
    end
    
    for f = 1:length(wav_files)
        file_name = wav_files(f).name;
        input_file = fullfile(ds.path, file_name);
        
        % Extract the button name from the file (e.g., '9.wav' becomes '9')
        [~, button_name, ~] = fileparts(file_name);
        
        fprintf('Segmenting Key: [%s] | Source: %s\n', button_name, ds.name);
        
        % Read physical audio
        [x, Fe] = audioread(input_file);
        x = mean(x, 2); % Mono conversion
        
        %% 3. Peak Detection Logic
        abs_x = abs(x);
        triggers = find(abs_x > ds.threshold);
        seg_matrix = [];
        last_sample = 0;
        
        for i = 1:length(triggers)
            current_sample = triggers(i);
            
            % Prevent overlap: ensure minimum distance between keystrokes
            if current_sample > last_sample
                t_start = (current_sample / Fe) - ds.pre_trigger;
                t_end = t_start + ds.segment_duration;
                
                seg_matrix = [seg_matrix; t_start, t_end];
                last_sample = current_sample + (ds.min_dist_seconds * Fe);
            end
        end
        
        %% 4. Export the Segments
        if isempty(seg_matrix)
            fprintf('  -> NO SEGMENTS FOUND! (Check your threshold)\n');
            continue;
        end
        
        % Create the specific output folder for this key
        output_folder = fullfile(output_base_folder, button_name);
        if ~exist(output_folder, 'dir')
            mkdir(output_folder);
        end
        
        session_time = datestr(now, 'yyyymmdd_HHMMSS');
        
        for i = 1:size(seg_matrix, 1)
            t_start = seg_matrix(i, 1);
            t_end = seg_matrix(i, 2);
            
            % Automatically tag the filename with the button and laptop brand
            % e.g. "9_HP_20260426_143000_01.wav"
            out_filename = sprintf('%s_%s_%s_%02d.wav', button_name, ds.name, session_time, i);
            
            start_sample = max(1, round(t_start * Fe));
            end_sample = min(length(x), round(t_end * Fe));
            
            % Save to disk
            audiowrite(fullfile(output_folder, out_filename), x(start_sample:end_sample), Fe);
        end
        
        fprintf('  -> Success: Extracted %d keystrokes.\n', size(seg_matrix, 1));
    end
end

fprintf('\n=== BATCH SEGMENTATION 100%% COMPLETE ===\n');
