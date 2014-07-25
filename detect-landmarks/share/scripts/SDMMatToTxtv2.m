%
% Used to convert models generated with the '21072014' code from
% Matlab to a text format.
%
function [] = SDMMatToTxtv2()

filename = 'C:\Users\Patrik\Documents\GitHub\SDM_Zhenhua_21072014_ibugLfpwPatrik\output_CR.mat';
load(filename);

out = 'SDM_Model_HOG_Zhenhua_22072014.txt';
fout = fopen(out, 'w');

description = '# SDM Model with F-HOG features, trained by Patrik, 22.7.2014';
fprintf(fout, '%s\n', description);

numLandmarks = size(mean_shape, 1) / 2; % 15
fprintf(fout, 'numLandmarks %d\n', numLandmarks);

% output the landmark ids
landmarkIds = [18, 22, 23, 27, 31, 32, 36, 37, 40, 43, 46, 49, 55, 52, 58];
for i=1:numLandmarks
    fprintf(fout, '%d\n', landmarkIds(i));
end

% We do some processing to the mean:
% - we move it to [-0.5, 0.5] x [-0.5, 0.5] by moving the min/max
% - we scale it by sx, sy
% - we translate it by tx, ty (no relative factor involved because this is already in relation to the face box size)
meanModelWidth = max(mean_shape(1:numLandmarks)) - min(mean_shape(1:numLandmarks));
meanModelHeight = max(mean_shape(numLandmarks+1:end)) - min(mean_shape(numLandmarks+1:end));
faceboxWidth = meanModelWidth / mean_diff(3);
faceboxHeight = meanModelHeight / mean_diff(4); % they should both be the same, but they're not (zf?)
faceboxAvgDim = (faceboxWidth + faceboxHeight) / 2;

mean_shape(1:numLandmarks) = mean_shape(1:numLandmarks) - mean(mean_shape(1:numLandmarks));
mean_shape(numLandmarks+1:end) = mean_shape(numLandmarks+1:end) - mean(mean_shape(numLandmarks+1:end));

mean_shape = mean_shape / faceboxAvgDim;

mean_shape(1:numLandmarks) = mean_shape(1:numLandmarks) + mean_diff(1);
mean_shape(numLandmarks+1:end) = mean_shape(numLandmarks+1:end) + mean_diff(2);

for i=1:numLandmarks*2 % first 15 x-coordinates, then 15 y-coordinates
    fprintf(fout, '%f\n', mean_shape(i));
end

numCascadeSteps = size(CR, 2);
fprintf(fout, 'numCascadeSteps %d\n', numCascadeSteps);

% the parameters for each hog scale: {cellSize, numBins} (numBins = numOrientations)
%params = { {3, 4}, {3, 4}, {2, 4}, {2, 4}, {1, 4} };
descriptorType = 'vlhog-uoctti'; % = fhog
descriptorPostprocessing = 'none';
descriptorParameters = '';

for i=1:numCascadeSteps
    featureDimensionRows = size(CR(i).A, 1); % [r, c] = size(). 1 returns rows, 2 columns.
    featureDimensionCols = size(CR(i).A, 2);
    assert(featureDimensionCols/2 == numLandmarks, 'something wrong, stop!');
    fprintf(fout, 'cascadeStep %d rows %d cols %d\n', i-1, featureDimensionRows, featureDimensionCols);
    fprintf(fout, 'descriptorType %s\n', descriptorType);
    fprintf(fout, 'descriptorPostprocessing %s\n', descriptorPostprocessing);
    % descriptorParameters = sprintf('cellSize %d numBins %d', params{i}{1}, params{i}{2}); % for old static vlhog
    fprintf(fout, 'descriptorParameters %s\n', descriptorParameters);
    for r=1:featureDimensionRows
       for c=1:featureDimensionCols
           fprintf(fout, '%f ', CR(i).A(r, c));
       end
       fprintf(fout, '\n');
    end
end

fclose(fout);

end