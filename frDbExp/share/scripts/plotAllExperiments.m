clear;
close all;

plotCurves('C:\Users\Patrik\Documents\MATLAB\probes_pm15_fte.txt', 'r', 0, 1); % 1 = plot with FTEs, 1 = plotIndividualDataPoints
%plotCurves('C:\Users\Patrik\Documents\MATLAB\probes_pm30_fte.txt', 'b', 0, 1);
%plotCurves('C:\Users\Patrik\Documents\MATLAB\probes_pm45_fte.txt', 'g', 0, 1);
%plotCurves('C:\Users\Patrik\Documents\MATLAB\probes_pm60_fte.txt', 'y', 0, 1);

%plotCurves('C:\Users\Patrik\Documents\MATLAB\probes_all.txt', 'r', 0, 1); % add parameter verifRateAtFAR


% Todo:
% -- the numbers: 1) ident. rate 2) ver. rate at x% far (oä)
% -- bei 15°, die DET kurve bzw die vektoren sollten doch egtl 0 elemente
% enthalten, nur ein punkt bei 0, 0 ?