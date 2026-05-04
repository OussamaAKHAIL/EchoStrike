% ========================================================
% FULL KEYBOARD INFERENCE & 2D TOPOGRAPHIC VISUALIZATION
% ========================================================
% This script loads your final trained full keyboard model, grabs a 
% completely random audio clip from your segmented data, extracts it, 
% predicts what key it was, and then draws a massive, minimalistic QWERTY
% keyboard with a glowing green topographic heatmap indicating probabilities!

clc; clear; close all;

%% 1. Load the Best Model
fprintf('Loading Full Keyboard Model...\n');
if exist('trained_svm_full_keyboard.mat', 'file')
    load('trained_svm_full_keyboard.mat');
    model = svm_model;
else
    error('Could not find a trained model. Please run train_full_keyboard.m first!');
end

%% 2. Grab a Random Test Audio
% TODO: Replace with the path to your dynamically segmented audio folders
base_path = 'path/to/your/segmented_audios';
folders = dir(base_path);
folders = folders(~ismember({folders.name}, {'.', '..'}));

% Pick a random key category, then a random clip
rand_folder = folders(randi(length(folders))).name;
file_list = dir(fullfile(base_path, rand_folder, '*.wav'));
test_file_path = fullfile(base_path, rand_folder, file_list(randi(length(file_list))).name);

fprintf('\n=> Randomly selected a test clip for True Key: [%s]\n', rand_folder);

%% 3. Apply the Golden Processing Pipeline
[x, Fe] = audioread(test_file_path);
x = mean(x, 2); 
Te = 1/Fe;
N_segment = length(x);
f_segment = -Fe/2 : Fe/N_segment : (Fe/2 - Fe/N_segment);
X_segment = fftshift(fft(x) * Te);

pos_idx = find(f_segment >= 0);
fft_magnitude = abs(X_segment(pos_idx));

features_log = 20 * log10(fft_magnitude' + eps);
features_log = features_log - min(features_log);
features = features_log / (max(features_log) + eps);

test_feature = resample(features, 1000, length(features));

%% 4. Predict the Key!
[pred_class, score] = predict(model, test_feature);

if iscell(pred_class) || iscategorical(pred_class)
    pred_class_str = char(pred_class(1));
else
    pred_class_str = num2str(pred_class); 
end
class_names = model.ClassNames;

fprintf('=> The AI predicts this is the [%s] key!\n\n', pred_class_str);

%% 5. Define the QWERTY Keyboard Map
% We use containers.Map to map the exact text labels to physical coordinates
% Format: keys('keyname') = [x_center, y_center, width]
km = containers.Map();

% Row 5 (Numbers)
km('backtick()') = [1, 5, 1]; km('1') = [2, 5, 1]; km('2') = [3, 5, 1]; 
km('3') = [4, 5, 1]; km('4') = [5, 5, 1]; km('5') = [6, 5, 1]; 
km('6') = [7, 5, 1]; km('7') = [8, 5, 1]; km('8') = [9, 5, 1]; 
km('9') = [10, 5, 1]; km('0') = [11, 5, 1]; km('dash(-)') = [12, 5, 1]; 
km('equal(=)') = [13, 5, 1]; km('backspace') = [14.5, 5, 2];

% Row 4 (QWERTY)
km('tab') = [1.5, 4, 1.5]; km('q') = [3, 4, 1]; km('w') = [4, 4, 1]; 
km('e') = [5, 4, 1]; km('r') = [6, 4, 1]; km('t') = [7, 4, 1]; 
km('y') = [8, 4, 1]; km('u') = [9, 4, 1]; km('i') = [10, 4, 1]; 
km('o') = [11, 4, 1]; km('p') = [12, 4, 1]; km('bracketopen({)') = [13, 4, 1]; 
km('bracketclose(})') = [14, 4, 1]; km('backslash') = [15, 4, 1.5];

% Row 3 (ASDF)
km('caps') = [1.75, 3, 1.75]; km('a') = [3.25, 3, 1]; km('s') = [4.25, 3, 1]; 
km('d') = [5.25, 3, 1]; km('f') = [6.25, 3, 1]; km('g') = [7.25, 3, 1]; 
km('h') = [8.25, 3, 1]; km('j') = [9.25, 3, 1]; km('k') = [10.25, 3, 1]; 
km('l') = [11.25, 3, 1]; km('semicolon(;)') = [12.25, 3, 1]; 
km('apostrophe('')') = [13.25, 3, 1]; km('enter') = [14.8, 3, 2];

% Row 2 (ZXCV)
km('Lshift') = [2.25, 2, 2.25]; km('LShift') = [2.25, 2, 2.25];
km('z') = [4, 2, 1]; km('x') = [5, 2, 1]; km('c') = [6, 2, 1]; 
km('v') = [7, 2, 1]; km('b') = [8, 2, 1]; km('n') = [9, 2, 1]; 
km('m') = [10, 2, 1]; km('comma(,)') = [11, 2, 1]; km('fullstop') = [12, 2, 1]; 
km('slash') = [13, 2, 1]; km('Rshift') = [14.7, 2, 2.5]; 

% Row 1 (Spacebar)
km('Lctrl') = [1.5, 1, 1.5]; km('cmd') = [2.5, 1, 1.2]; 
km('Lalt') = [3.5, 1, 1.2]; km('space') = [7.5, 1, 6]; 
km('Ralt') = [11.5, 1, 1.2]; km('altR') = [11.5, 1, 1.2]; 
km('fn') = [12.5, 1, 1.2]; km('menu') = [13.5, 1, 1.2]; 
km('Rctrl') = [14.5, 1, 1.5];

% Arrow Keys & Navigation
km('Left Arrow') = [16.5, 1, 1]; km('left') = [16.5, 1, 1];
km('Right Arrow') = [18.5, 1, 1]; km('right') = [18.5, 1, 1];
km('Up Arrow') = [17.5, 2, 1]; km('up') = [17.5, 2, 1];
km('Down Arrow') = [17.5, 1, 1]; km('down') = [17.5, 1, 1];

km('esc') = [1, 6, 1]; km('asterisk') = [11, 6, 1];
km('home') = [17.5, 6, 1]; km('end') = [18.5, 6, 1];
km('pgup') = [18.5, 5, 1]; km('pg up') = [18.5, 5, 1];
km('pgdn') = [18.5, 4, 1]; km('pg dn') = [18.5, 4, 1];

%% 6. Generate the Topographic Heatmap (Z Matrix)
grid_width = 20; grid_height = 6;
Z = zeros(grid_height, grid_width);

% SVM models output unbounded (negative) margin scores rather than absolute percentages.
% To prevent topographic contour errors, we mathematically force all scores into a bounded [0, 100] scale.
score_shifted = score - min(score);
if max(score_shifted) > 0
    calculated_probs = (score_shifted / max(score_shifted)) * 100;
else
    calculated_probs = zeros(size(score)); 
end

% Inject probabilities into physical locations
for i = 1:length(class_names)
    c_name = char(class_names(i));
    if isKey(km, c_name)
        prob = calculated_probs(1, i);
        coords = km(c_name);
        
        xc = round(coords(1)); yc = round(coords(2));
        
        % Ensure bounds just in case
        if xc > 0 && xc <= grid_width && yc > 0 && yc <= grid_height
            Z(yc, xc) = max(Z(yc, xc), prob); % Store highest match
        end
    end
end

%% 7. Visualize!
figure('Name', 'Full Keyboard Acoustic Prediction', 'Position', [50, 100, 1200, 450], 'Color', [1 1 1]);

% Make the Heatmap Interpolation
[X, Y] = meshgrid(1:grid_width, 1:grid_height);
[Xq, Yq] = meshgrid(linspace(0.5, grid_width+0.5, 300), linspace(0.5, grid_height+0.5, 100));
Zq = interp2(X, Y, Z, Xq, Yq, 'spline');
% Ensure zero-probability zones remain perfectly flat 
Zq(Zq < 0) = 0; 

% Draw the Topographic Mountain Lines
ax = axes('Position', [0.05 0.05 0.9 0.85]);
hold on;
contourf(Xq, Yq, Zq, 30, 'LineColor', [0.2 0.8 0.2], 'LineWidth', 0.5); % Brighter topography lines

% =====================================================
% CHANGE THE HEATMAP COLORS HERE:
% [Red, Green, Blue] values from 0.0 (Black) to 1.0 (Bright)
color_cold = [0.0, 0.2, 0.0]; % 0% Probability color (Darker Green)
color_hot  = [0.4, 1.0, 0.4]; % 100% Probability color (Neon Lime Glow)
% =====================================================

% Generate custom gradient colormap
c_levels = 256;
custom_neon_green = [linspace(color_cold(1), color_hot(1), c_levels)', ...
                     linspace(color_cold(2), color_hot(2), c_levels)', ...
                     linspace(color_cold(3), color_hot(3), c_levels)'];
colormap(custom_neon_green);

% Overlay the Minimalist Keyboard Outline
keyboard_keys = keys(km);
for k = 1:length(keyboard_keys)
    k_name = keyboard_keys{k};
    coords = km(k_name);
    xc = coords(1); yc = coords(2); w = coords(3);
    h = 1; % Assume height of 1
    
    % Draw minimalistic rectangle over the topography
    rectangle('Position', [xc - w/2, yc - h/2, w, h], ...
              'EdgeColor', [1 1 1], 'LineWidth', 1.5, 'FaceColor', 'none', 'Curvature', 0.1);
              
    % Add the Text label to it
    clean_k_name = strrep(k_name, '()', '');
    clean_k_name = strrep(clean_k_name, '(', '');
    clean_k_name = strrep(clean_k_name, ')', '');
    text(xc, yc, clean_k_name, 'FontSize', 8, 'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', 'Color', [1 1 1], 'Interpreter', 'none');
end

axis equal off;
xlim([0, 19.5]); ylim([0.5, 6.5]);

% Title Result
title_txt = sprintf('AI Prediction: [%s]  ---  True Physical Key: [%s]', pred_class_str, rand_folder);
annotation('textbox', [0 0.9 1 0.1], 'String', title_txt, 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', 'black', 'FontWeight', 'bold');

hold off;
fprintf('Open the plot to view the real-time acoustic topographic map!\n');
