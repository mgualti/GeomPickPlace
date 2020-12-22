function CombineResultsPacking()

    %% Parameters
    
    file1 = '../results-packing-isec10-test1-gqMc-200.mat';
    file2 = '../results-packing-isec10-test1-gqMc-30.mat';
    outFile = '../results-packing-isec10-test1-gqMc.mat';
    
    timeFactor1 = 1.0;
    timeFactor2 = 1.0;
    
    %% Load data and combine
    
    data1 = load(file1);
    data2 = load(file2);
    out = struct();
    
    % parameters
    out.params = data1.params;
    out.nObjects = data1.nObjects;
    
    % results
    out.nPlaced = [data1.nPlaced, data2.nPlaced];
    out.planLength = [data1.planLength, data2.planLength];
    out.graspSuccess = [data1.graspSuccess, data2.graspSuccess];
    out.graspAntipodal = [data1.graspAntipodal, data2.graspAntipodal];
    out.tempPlaceStable = [data1.tempPlaceStable, data2.tempPlaceStable];
    out.packingHeight = [data1.packingHeight, data2.packingHeight];
    out.taskPlanningTime = [timeFactor1 * data1.taskPlanningTime, timeFactor2 * data2.taskPlanningTime];
    out.regraspPlanningTime = [timeFactor1 * data1.regraspPlanningTime, timeFactor2 * data2.regraspPlanningTime];
    out.totalTime = timeFactor1 * data1.totalTime + timeFactor2 * data2.totalTime;
    
    %% Save result
    
    save(outFile, '-struct', 'out');