function TestSignificanceOfResults()
    
    %% Input Data
    
%     baselineSuccesses = 151 + 150;
%     baselineAttempts = 180 + 180;
%     treatmentSuccesses = 164 + 165;
%     treatmentAttempts = 180 + 180;
    
    baselineSuccesses = 173 + 174;
    baselineAttempts = 196 + 201;
    treatmentSuccesses = 196 + 196;
    treatmentAttempts = 207 + 210;

    %% Student's t-test
    
    % Null hypothesis: The treatment resulted in no improvement over the baseline.
    % Alternative hypothesis: The treatment resulted in a statistically significant improvement.
    
    successB = [ones(1, baselineSuccesses), zeros(1, baselineAttempts - baselineSuccesses)];
    successT = [ones(1, treatmentSuccesses), zeros(1, treatmentAttempts - treatmentSuccesses)];
    [h, p] = ttest2(successB, successT, 'tail', 'left');
    if h, isSig = 'is'; else isSig = 'is not'; end
    disp(['Success rate ' isSig ' significantly greater than baseline (p = ' num2str(p) ').']);
    