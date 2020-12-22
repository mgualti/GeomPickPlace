function PlotResultsCanonical()

    %% Parameters
    
    %resultsFileName = '../results-canonical-isec11-test1-none.mat';
    %resultsFileName = '../results-canonical-isec11-test1-step.mat';
    %resultsFileName = '../results-canonical-ur5-test1-gq.mat';
    resultsFileName = '../results-canonical-ur5-test1-cu.mat';
    %resultsFileName = '../results-canonical-isec11-test1-sp.mat';
    %resultsFileName = '../results-canonical-isec11-test1-mc.mat';
    %resultsFileName = '../results-canonical-isec10-test1-gqMc.mat';
    
    if contains(resultsFileName, 'isec11'), timeFactor = 0.90; else, timeFactor = 1.0; end
    
    fmt = '%0.3f';
    %pm = [' ' char(177) ' '];
    pm = ' $\pm$ ';

    %% Load

    close('all');

    if ~exist(resultsFileName, 'file')
        disp([resultsFileName ' not found.']);
        return
    end
    load(resultsFileName);
    disp(' ');
    
    nEpisodes = length(nPlaced);
    disp(['Completed ' num2str(nEpisodes) ' episodes.']);
    
    %% Task Completion Related
    
    placeSuccess = [ones(1, sum(nPlaced)), zeros(1, nEpisodes - sum(nPlaced))];
    avgPlaceSuccess = mean(placeSuccess);
    errPlaceSuccess = std(placeSuccess) / sqrt(length(placeSuccess));
    disp(['Place success rate: ' num2str(avgPlaceSuccess, fmt) pm num2str(errPlaceSuccess, fmt)])
    
    nRegraspPlans = length(planLength);
    execSuccess = [ones(1, sum(nPlaced)), zeros(1, nRegraspPlans - sum(nPlaced))];
    avgExecSuccess = mean(execSuccess);
    errExecSuccess = std(execSuccess) / sqrt(length(execSuccess));
    disp(['Place execution success rate: ' num2str(avgExecSuccess, fmt) pm num2str(errExecSuccess, fmt)]);
    
    nRegraspPlanAttempts = length(regraspPlanningTime);
    regraspPlanFound = [ones(1, nRegraspPlans), zeros(1, nRegraspPlanAttempts - nRegraspPlans)];
    avgRegraspPlanFound = mean(regraspPlanFound);
    errRegraspPlanFound = std(regraspPlanFound) / sqrt(length(regraspPlanFound));
    disp(['Regrasp plan found rate: ' num2str(avgRegraspPlanFound, fmt) pm num2str(errRegraspPlanFound, fmt)]);
    
    graspSuccessRate = num2str(mean(graspSuccess), fmt);
    errGraspSuccessRate = num2str(std(graspSuccess) / sqrt(length(graspSuccess)), fmt);
    disp(['Grasp success rate: ' graspSuccessRate pm errGraspSuccessRate]);
    
    graspAntipodalRate = num2str(mean(graspAntipodal), fmt);
    errGraspAntipodalRate = num2str(std(graspAntipodal) / sqrt(length(graspAntipodal)), fmt);
    disp(['Grasp antipodal rate: ' graspAntipodalRate pm errGraspAntipodalRate]);
    
    placeStableRate = num2str(mean(tempPlaceStable), fmt);
    errPlaceStableRate = num2str(std(tempPlaceStable) / sqrt(length(tempPlaceStable)), fmt);
    disp(['Temporary placement stability rate: ' placeStableRate pm errPlaceStableRate]);
    
    avgPlanLength = num2str(mean(planLength), fmt);
    errPlanLength = num2str(std(double(planLength)) / sqrt(length(planLength)), fmt);
    nPlanLengthsGreaterThan2 = num2str(sum(planLength > 2));
    nPlanLengths = num2str(length(planLength));
    disp(['Average plan length: ' avgPlanLength pm errPlanLength]);
    disp(['Number of plan lengths greater than 2: ' nPlanLengthsGreaterThan2 ' / ' nPlanLengths]);
    
    %% Planning Times
    
    taskPlanningTime = timeFactor * taskPlanningTime;
    avgTaskPlanningTime = num2str(mean(taskPlanningTime), fmt);
    errTaskPlanningTime = num2str(std(taskPlanningTime) / sqrt(length(taskPlanningTime)), fmt);
    disp(['Average task planning time: ' avgTaskPlanningTime pm errTaskPlanningTime ' (s)']);
    
    regraspPlanningTime = timeFactor * regraspPlanningTime;
    avgRegraspPlanningTime = num2str(mean(regraspPlanningTime), fmt);
    errRegraspPlanningTime = num2str(std(regraspPlanningTime) / sqrt(length(regraspPlanningTime)), fmt);
    disp(['Average regrasp planning time: ' avgRegraspPlanningTime pm errRegraspPlanningTime ' (s)']);
    
    totalTime = timeFactor * totalTime;
    totalTime = num2str(totalTime / 3600, fmt);
    disp(['Total time: ' totalTime ' (h)']);