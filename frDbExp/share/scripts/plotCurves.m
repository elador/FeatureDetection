function [] = plotCurves(filename, curveColor, plotWithFtes, plotIndividualDataPoints)


%filename = 'C:\Users\Patrik\Documents\MATLAB\result.txt';
[experimentTitle, curveLabels, numFte, positives, negatives] = readFpFnCurve(filename);

% Stuff the FTEs to the right place
if (plotWithFtes == 1)
    positives_y = 0:(1/double(size(positives, 2)+numFte - 1)):1;
    numFteVec = zeros(1, numFte);
    fte_yAxisVal = double(numFte)/double(size([numFteVec, positives], 2));
    fte_yAxisValVec = fte_yAxisVal * ones(1, numFte);
    positives_y(1:numFte) = fte_yAxisValVec;
    positives_with_fte = [numFteVec, positives];
else
    positives_y = 0:(1/double(size(positives, 2) - 1)):1;
    positives_with_fte = positives;
end

ftePercentage = (double(numFte) / double(numFte+size(positives, 2))) * 100;

% Plot the FN-FP curves for the negative and positive scores
figure(1);
if (plotIndividualDataPoints == 1)
    plot(positives_with_fte, positives_y, [curveColor, '-o'], 'DisplayName', ['FRR (positives) ' curveLabels]);
else
    plot(positives_with_fte, positives_y, [curveColor, '-'], 'DisplayName', ['FRR (positives) ' curveLabels]);
end
legend('-DynamicLegend');
title(['\fontsize{10}Cumulative histogram of the scores of the positive and negative ground-truth', '\newline\fontsize{9}\color{red}\it{' experimentTitle '}']);
xlabel('Score');
ylabel('Frequency');
hold on;
negatives_y = 0:(1/(size(negatives, 2) - 1)):1;
if (plotIndividualDataPoints == 1)
    plot(negatives, negatives_y, [curveColor, '--x'], 'DisplayName', ['FAR (negatives) ' curveLabels]);
else
    plot(negatives, negatives_y, [curveColor, '--'], 'DisplayName', ['FAR (negatives) ' curveLabels]);
end

%if (plotIndividualDataPoints == 1)
%    plot(positives_with_fte, positives_y, [curveColor, 'o'], 'DisplayName', ['FRR (positives) ' curveLabels]);
%    plot(negatives, negatives_y, [curveColor, 'x'], 'DisplayName', ['FAR (negatives) ' curveLabels]);   
%end

% Calculate and plot the DET curve
figure(2);
[det_fp, det_fn, det_scores] = fpfn2detCurve(positives_with_fte, positives_y, negatives, negatives_y);

% Calculate the EER
[point1_fn_val_idx, point2_fn_val_idx, eer] = calcEER(positives_with_fte, positives_y, negatives, negatives_y, det_fp, det_fn, det_scores);

%plot(det_fp, det_fn, [curveColor, '-'], 'DisplayName', [experimentTitle, '; FTE: ', num2str(ftePercentage), '%']);
plot(det_fp, det_fn, [curveColor, '-'], 'DisplayName', [experimentTitle, '; FTE: ', num2str(ftePercentage), '%', '; EER: ', num2str(eer*100), '%']);
set(gca,'xscale','log');
legend('-DynamicLegend');
title(['\fontsize{10}DET curve', '\newline\fontsize{9}\color{red}\it{' experimentTitle '}']);
xlabel('False positives (FAR)');
ylabel('False negatives (FRR)');
hold on;


% % TODO Show some score-values
% % TODO calculate the EER
% for i=1:30:size(det_fp, 2)
%     text(det_fp(i), det_fn(i), ['\bullet\leftarrow\fontname{times}',num2str(det_scores(i))],'FontSize',14)
% end
% hold on
% for i=1:4:32
%     text(det_fp(i), det_fn(i), ['\bullet\leftarrow\fontname{times}',num2str(det_scores(i))],'FontSize',14)
% end
% i=2;
% text(det_fp(i), det_fn(i), ['\bullet\leftarrow\fontname{times}',num2str(det_scores(i))],'FontSize',14)
% i=3;
% text(det_fp(i), det_fn(i), ['\bullet\leftarrow\fontname{times}',num2str(det_scores(i))],'FontSize',14)
% i=4;
% text(det_fp(i), det_fn(i), ['\bullet\leftarrow\fontname{times}',num2str(det_scores(i))],'FontSize',14)
% 
% eer_best_difference = Inf;
% eer_best_thresh = 0;
% for i=1:size(det_fn, 2)
%     if (abs(det_fn(i)-det_fp(i)) < eer_best_difference)
%         eer_best_difference = abs(det_fn(i)-det_fp(i));
%         eer_best_thresh = det_scores(i);
%         eer_best_percent = det_fp(i);
%         eer_best_percent_fn = det_fn(i); % not necessary, should be equal, only for checking purposes
%     end
% end

end