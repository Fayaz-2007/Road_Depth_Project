clc;
clear;

% Load trained model
load('final_regression_model.mat', 'model');

% Select image manually
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select Road Image');
imgPath = fullfile(path, file);

img = imread(imgPath);

if size(img,3) == 3
    imgGray = rgb2gray(img);
else
    imgGray = img;
end

imgGray = imresize(imgGray, [128 128]);

% --- Feature Extraction (same as training) ---

f1 = mean(imgGray(:));
f2 = std(double(imgGray(:)));
edges = edge(imgGray, 'Canny');
f3 = sum(edges(:));
f4 = entropy(imgGray);
f5 = max(imgGray(:));
f6 = min(imgGray(:));

glcm = graycomatrix(imgGray,'Offset',[0 1]);
stats = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});

f7 = stats.Contrast;
f8 = stats.Correlation;
f9 = stats.Energy;
f10 = stats.Homogeneity;

featureVector = double([f1 f2 f3 f4 f5 f6 f7 f8 f9 f10]);

% Predict displacement
predictedValue = predict(model, featureVector);

% Display result
figure;
imshow(img);
title(sprintf('Predicted Displacement: %.2f cm', predictedValue));

fprintf('Predicted Displacement: %.2f cm\n', predictedValue);

