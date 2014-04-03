%[diff10] = landmarkEvaluation('C:\\Users\\Patrik\\Documents\\GitHub\\sdm_lfpw_tr_10lm_5s_3c_RESULTS.txt');
%[diff34] = landmarkEvaluation('C:\\Users\\Patrik\\Documents\\GitHub\\sdm_lfpw_tr_34lm_10s_5c_RESULTS.txt');
%[diff68] = landmarkEvaluation('C:\\Users\\Patrik\\Documents\\GitHub\\sdm_lfpw_tr_68lm_10s_5c_RESULTS.txt');

[diff68] = landmarkEvaluation('C:\\Users\\Patrik\\Documents\\GitHub\\sdmEvalOut\\results.txt');


figure(1);
%plot(sort(diff10), [0:1/(length(diff10)-1):1], 'r');
hold on;
%plot(sort(diff34), [0:1/(length(diff34)-1):1], 'g');
plot(sort(diff68), [0:1/(length(diff68)-1):1], 'b');
title('');
title(['\fontsize{10} Landmark detection accuracy on LFPW, evaluated on 10 points ', ...
       '\newline \fontsize{8} \it The model is trained on different LMs but always evaluated on the same 10 LMs. ']);
xlabel 'Error normalized by IED';
ylabel 'Data percentage';
%legend(sprintf('10lms, mean=%1.4f', mean(diff10)), sprintf('34lms, mean=%1.4f', mean(diff34)), sprintf('68lms, mean=%1.4f', mean(diff68)));
legend(sprintf('20 lms, mean=%1.4f', mean(diff68)));
%find(diff>0.3)

%mean(diff10)
%std(diff10)
%mean(diff34)
%std(diff34)
mean(diff68)
std(diff68)
