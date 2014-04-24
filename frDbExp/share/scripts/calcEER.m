function [point1_fn_val_idx, point2_fn_val_idx, eer] = calcEER(pos_x, pos_y, neg_x, neg_y, det_fp, det_fn, det_scores)

% Some useful code-snippets:
% http://www.cs.cmu.edu/~keystroke/evaluation-script.R

% We could maybe improve the script by finding the 2 points, then loop
% through the points on the other curve (the one with more points) and grab
% the value there.

% Find the 2 points on the curve slightly below/above the EER-point
dists = det_fp - det_fn;
idx_pos = find(dists>=0);
% I think ideally we should always find a point here. I think
% fpfn2detCurve() is handling an edge-case not so well and 'det_fn' should
% go to 0? When looping through the threshs, it should always take the
% lowest value, not the highest
if isempty(idx_pos) % if we don't find a positive point, the EER is always 0?
    point1_fn_val_idx = 0;
    point2_fn_val_idx = 0;
    eer = 0;
    return;
end    
pos_min_val = min(dists(idx_pos));
pos_min_val_idx = find(dists == pos_min_val);

% The index of the point in the positive values that is used for the calculation
point1_fn_val_idx = find(pos_y == det_fn(pos_min_val_idx));

idx_neg = find(dists<0);
neg_min_val = max(dists(idx_neg));
neg_min_val_idx = find(dists == neg_min_val);

% The index of the point in the positive values that is used for the calculation
point2_fn_val_idx = find(pos_y == det_fn(neg_min_val_idx));


p1 = [det_fp(pos_min_val_idx), det_fn(pos_min_val_idx)];
p2 = [det_fp(neg_min_val_idx), det_fn(neg_min_val_idx)];

% Extract the two points as (x) and (y), and find the point on the
% line between x and y where the first and second elements of the
% vector are equal.  Specifically, the line through x and y is:
%   x + a*(y-x) for all a, and we want a such that
%   x[1] + a*(y[1]-x[1]) = x[2] + a*(y[2]-x[2]) so
%   a = (x[1] - x[2]) / (y[2]-x[2]-y[1]+x[1])
a = ( p1(1) - p1(2) ) / ( p2(2) - p1(2) - p2(1) + p1(1) );
eer = p1(1) + a * ( p2(1) - p1(1) );

s1 = det_scores(pos_min_val_idx);
s2 = det_scores(neg_min_val_idx);

interp_score = (1-a)*s1 + (a)*s2; % correct?

% Plot the points (debug output)
figure(2);
plot(p1(1), p1(2), 'b+', 'DisplayName', 'eer positive point 1');
plot(p2(1), p2(2), 'r+', 'DisplayName', 'eer positive point 2');
plot([0, 1], [0, 1], 'g-', 'DisplayName', 'eer line');
plot(eer, eer, 'm+', 'DisplayName', 'eer point');

figure(1);
plot(pos_x(point1_fn_val_idx), pos_y(point1_fn_val_idx), 'bo', 'DisplayName', 'eer positive point 1');
plot(pos_x(point2_fn_val_idx), pos_y(point2_fn_val_idx), 'ro', 'DisplayName', 'eer positive point 2');
plot(interp_score, eer, 'mo', 'DisplayName', 'eer point');

figure(2);

end