% pos_x and pos_y have to have the same length, the same goes for neg_x/y.
% Handling of FTE's: We just draw, if several threshs have the same value,
% we take only the best one.

function [det_fp, det_fn, det_scores] = fpfn2detCurve(pos_x, pos_y, neg_x, neg_y)

det_fp = zeros(1, size(pos_x, 2));
det_fn = zeros(1, size(pos_x, 2));
det_scores = zeros(1, size(pos_x, 2));

currentThresh = 0;
currentIndexInPositives = 1;

for i=1:size(pos_x, 2)
    
    pos_this_x_score = pos_x(i); % the score = the moving thresh, we're moving through the pos=FRR=FN curve
    pos_this_y_freq = pos_y(i); % the frequency = the FN=FRR
    
    if (i==1)
        previousThresh = pos_this_x_score;
    else
        previousThresh = currentThresh;
    end
    currentThresh = pos_this_x_score;
    
    % Handling of special cases when there are no FP-points at the start or end of the fp-curve
    hist_upper_interp_points_idx_FP = find(neg_x>=currentThresh);
    if (size(hist_upper_interp_points_idx_FP, 2) == 0)
        fp_upper_x = 1; % lower-right corner of fp-curve
        fp_upper_y = 0; % 
    else
        fp_upper_x = neg_x(hist_upper_interp_points_idx_FP(end));
        fp_upper_y = neg_y(hist_upper_interp_points_idx_FP(end));
    end
    
    hist_lower_interp_points_idx_FP = find(neg_x<currentThresh);
    if (size(hist_lower_interp_points_idx_FP, 2) == 0)
        fp_lower_x = 0;
        fp_lower_y = 1; % left-upper corner, the neg-curve always ends there

    else
        fp_lower_x = neg_x(hist_lower_interp_points_idx_FP(1));
        fp_lower_y = neg_y(hist_lower_interp_points_idx_FP(1));
    end
    
    % We could check if fn_lower and fn_upper are too close in x-dir, and if yes, don't interpolate and use the value directly.
    fp_true_y = fp_lower_y + (fp_upper_y - fp_lower_y) * ( (pos_this_x_score - fp_lower_x) / (fp_upper_x - fp_lower_x) ); % division by zero?
    
    if (previousThresh~=currentThresh)
        currentIndexInPositives = currentIndexInPositives + 1;
    end
    det_fn(currentIndexInPositives) = pos_this_y_freq; % the pos curve = FRR = FN
    det_fp(currentIndexInPositives) = fp_true_y; % the neg curve = FAR = FP
    det_scores(currentIndexInPositives) = currentThresh;
    
end

det_fp = det_fp(1:currentIndexInPositives);
det_fn = det_fn(1:currentIndexInPositives);
det_scores = det_scores(1:currentIndexInPositives);

end