function [experimentTitle, curveLabels, numFte, positives, negatives] = readFpFnCurve(filename)

fid = fopen(filename);
if (fid == -1)
    disp 'Error opening the file!'
end

experimentTitle = fgetl(fid);
curveLabels = fgetl(fid);
fteLine = fgetl(fid);
posLine = fgetl(fid);
negLine = fgetl(fid);

fteLineParsed = textscan(fteLine, '%s %d');
numFte = fteLineParsed{2};

posLineParsed = textscan(posLine, '%f', 'Delimiter', ' ');
positives = cell2mat(posLineParsed)';

negLineParsed = textscan(negLine, '%f', 'Delimiter', ' ');
negatives = cell2mat(negLineParsed)';

end