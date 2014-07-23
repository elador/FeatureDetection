function [] = SDMMatToTxt()

filename = 'output_CR.mat';
load(filename);

out = 'SDM_Model_HOG_Zhenhua_22072014.txt';
fout = fopen(out, 'w');

description = '# SDM Model with F-HOG features, trained by Patrik, 22.7.2014';
fprintf(fout, '%s\n', description);

numLandmarks = size(AAM.Mean_Landmark, 1) / 2; % 44
fprintf(fout, 'numLandmarks %d\n', numLandmarks);

for i=1:numLandmarks*2 % first 22 x-coordinates, then 22 y-coordinates
    fprintf(fout, '%f\n', AAM.Mean_Landmark(i));
end

numHogScales = size(AAM.RF.Regressor, 2);
fprintf(fout, 'numHogScales %d\n', numHogScales);

% the parameters for each hog scale: {cellSize, numBins} (numBins = numOrientations)
params = { {3, 4}, {3, 4}, {2, 4}, {2, 4}, {1, 4} };

for i=1:numHogScales
    featureDimensionRows = size(AAM.RF.Regressor(i).A, 1); % [r, c] = size(). 1 returns rows, 2 columns.
    featureDimensionCols = size(AAM.RF.Regressor(i).A, 2);
    assert(featureDimensionCols/2 == numLandmarks, 'something wrong, stop!');
    fprintf(fout, 'scale %d rows %d cols %d cellSize %d numBins %d\n', i-1, featureDimensionRows, featureDimensionCols, params{i}{1}, params{i}{2});
    
    for r=1:featureDimensionRows
       for c=1:featureDimensionCols
           fprintf(fout, '%f ', AAM.RF.Regressor(i).A(r, c));
       end
       fprintf(fout, '\n');
    end
end

fclose(fout);

end