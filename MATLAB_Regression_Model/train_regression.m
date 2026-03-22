clc;
clear;

%% Load Labels
labels = readtable(fullfile('..', 'dataset', 'labels.xlsx'));
numSamples = height(labels);
fprintf('Total Samples: %d\n', numSamples);

%% Feature Extraction
numFeatures = 10;   % 6 basic + 4 GLCM
X = zeros(numSamples, numFeatures);
Y = labels.displacement_cm;

for i = 1:numSamples
    
    % Search across category folders for the image
    imgName = labels.image_name{i};
    categoryDirs = {'bump', 'potholes', 'road'};
    imgPath = '';
    for c = 1:length(categoryDirs)
        candidate = fullfile('..', 'dataset', categoryDirs{c}, imgName);
        if isfile(candidate)
            imgPath = candidate;
            break;
        end
    end
    if isempty(imgPath)
        fprintf('Skipping %s (not found)\n', imgName);
        continue;
    end
    img = imread(imgPath);
    
    if size(img,3) == 3
        imgGray = rgb2gray(img);
    else
        imgGray = img;
    end
    
    imgGray = imresize(imgGray, [128 128]);
    
    % --- Basic Features ---
    f1 = mean(imgGray(:));
    f2 = std(double(imgGray(:)));
    edges = edge(imgGray, 'Canny');
    f3 = sum(edges(:));
    f4 = entropy(imgGray);
    f5 = max(imgGray(:));
    f6 = min(imgGray(:));
    
    % --- GLCM Texture Features ---
    glcm = graycomatrix(imgGray,'Offset',[0 1]);
    stats = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});
    
    f7 = stats.Contrast;
    f8 = stats.Correlation;
    f9 = stats.Energy;
    f10 = stats.Homogeneity;
    
    X(i,:) = [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10];
    
end

disp("Feature extraction completed.");

%% Train-Test Split
cv = cvpartition(numSamples,'HoldOut',0.2);

XTrain = X(training(cv),:);
YTrain = Y(training(cv));

XTest = X(test(cv),:);
YTest = Y(test(cv));

%% Boosted Ensemble Regression (Best Version)
model = fitrensemble(XTrain, YTrain, ...
    'Method','LSBoost', ...
    'NumLearningCycles',100);

disp("Model training completed.");

%% Prediction
YPred = predict(model, XTest);

%% Evaluation
mae = mean(abs(YPred - YTest));
rmse = sqrt(mean((YPred - YTest).^2));

fprintf('MAE: %.4f cm\n', mae);
fprintf('RMSE: %.4f cm\n', rmse);
save('final_regression_model.mat','model');
disp("Model saved successfully.");
