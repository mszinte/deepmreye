function expDes = designConfig(scr, const)
% ----------------------------------------------------------------------
% expDes = designConfig(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Define experimental design
% ----------------------------------------------------------------------
% Input(s) :
% scr : struct of the screen settings
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% expDes : struct containg experimental design
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Experimental condition
% 01 - intertrial interval
% 02 - fixation task
% 03 - pursuit task
% 04 - freeview task

% Experimental variables
% Var 1: fixation location
expDes.oneV = [1:1:const.fixations_postions]';
expDes.nb_var1 = length(expDes.oneV);
% 01 02 03 04 05
% 06 07 08 09 10
% 11 12 13 14 15
% 16 17 18 19 20
% 21 22 23 24 25

% Var 2: pursuit amplitude
expDes.twoV = [1:1:const.pursuit_amps]';
expDes.nb_var2 = length(expDes.twoV);
% 01: 3 dva/s
% 02: 5 dva/s
% 03: 7 dva/s

% Var 3: pursuit angle (nan first as defined later)
expDes.threeV = const.pursuit_angles';
expDes.nb_var3 = length(expDes.threeV);
% 01 = 0 deg
% 02 = 20 deg
% ...
% 18 = 340 deg

% Var 4: picture id number
expDes.fourV = [1:1:const.freeview_pics]';
expDes.nb_var4 = length(expDes.fourV);
% pic 01
% ... 
% pic 10

% Experimental loop
expDes.nb_var = 4;

% Fixation experimental loop
trial_repeat = 1;
while trial_repeat
    ii = 0;
    trialMat_fixation = zeros(const.nb_trials_fixation, expDes.nb_var+1)*nan;
    for rep = 1:const.nb_repeat_fixation
        for var1 = 1:expDes.nb_var1
            ii = ii + 1;
            trialMat_fixation(ii, 1) = 2;
            trialMat_fixation(ii, 2) = var1;
        end
    end
    trialMat_fixation = trialMat_fixation(randperm(const.nb_trials_fixation),:);
    trial_repeat = sum(diff(trialMat_fixation(:,2))==0);
end
trialMat_fixation = [1, 13, nan, nan, nan; % add intertrial interval
                     trialMat_fixation];    

% Pursuit experimental loop
ii = 0;
trialMat_pursuit = zeros(const.nb_trials_pursuit, expDes.nb_var+1)*nan;
for rep = 1:const.nb_repeat_pursuit
    for var2 = 1:expDes.nb_var2
        for var3 = 1:expDes.nb_var3
            ii = ii + 1;
            trialMat_pursuit(ii, 1) = 3;
            trialMat_pursuit(ii, 3) = var2;
            trialMat_pursuit(ii, 4) = var3; % redefined after to keep the dot in the screen
        end
    end
end
trialMat_pursuit = trialMat_pursuit(randperm(const.nb_trials_pursuit),:);

% Compute angle (var3)
pursuit_coords_on = [];
pursuit_coords_off = [];
for trial_pursuit = 1:const.nb_trials_pursuit
    pursuit_amp = const.pursuit_amp(trialMat_pursuit(trial_pursuit,3));
    pursuit_angle = const.pursuit_angles(trialMat_pursuit(trial_pursuit,4));
    recompute = 1;
    while recompute == 1
        if trial_pursuit == 1
            pursuit_coord_on = [scr.x_mid, scr.y_mid];
            pursuit_coord_off = [scr.x_mid + pursuit_amp * cosd(pursuit_angle),...
                                 scr.y_mid + pursuit_amp * -sind(pursuit_angle)];
        elseif trial_pursuit == const.nb_trials_pursuit
            pursuit_coord_on = pursuit_coords_off(trial_pursuit-1, :);
            pursuit_coord_off = [scr.x_mid, scr.y_mid];
        else
            pursuit_coord_on = pursuit_coords_off(trial_pursuit-1, :);
            pursuit_coord_off = pursuit_coord_on + [pursuit_amp * cosd(pursuit_angle), ...
                                                    pursuit_amp * -sind(pursuit_angle)];
        end
        
        % if fixation point leaves calibration window select another angle
        if pursuit_coord_off(1) < scr.x_mid - const.window_size/2 || pursuit_coord_off(1) > scr.x_mid + const.window_size/2 || ...
                pursuit_coord_off(2) < scr.y_mid - const.window_size/2 || pursuit_coord_off(2) > scr.y_mid + const.window_size/2
            recompute = 1;
            rand_val = randperm(length(const.pursuit_angles));
            trialMat_pursuit(trial_pursuit, 4) = rand_val(1);
            pursuit_angle = const.pursuit_angles(rand_val(1));
        else
            recompute = 0;
        end
    end
    pursuit_coords_on = [pursuit_coords_on; pursuit_coord_on];
    pursuit_coords_off = [pursuit_coords_off; pursuit_coord_off];
end
trialMat_pursuit = [1, 13, nan, nan, nan; % add intertrial interval
                    trialMat_pursuit];


% Freeview experimental loop
ii = 0;
trialMat_freeview = zeros(const.nb_trials_freeview, expDes.nb_var+1)*nan;
for rep = 1:const.nb_repeat_freeview
    for var4 = 1:expDes.nb_var4
        ii = ii + 1;
        trialMat_freeview(ii, 1) = 4;
        trialMat_freeview(ii, 5) = var4;
    end
end
trialMat_freeview = trialMat_freeview(randperm(const.nb_trials_freeview),:);
trialMat_freeview = [1, 13, nan, nan, nan; ... % add intertrial interval
                     trialMat_freeview; ...
                     1, 13, nan, nan, nan; % add end interval
                     ];    

% Define main matrix
trialMat = [trialMat_fixation; ...
            trialMat_pursuit; ...
            trialMat_freeview];

expDes.expMat = [zeros(const.nb_trials,2)*nan, ...
    zeros(const.nb_trials,1)*0+const.runNum,...
    [1:const.nb_trials]',trialMat];

% 01 : onset
% 02 : duration
% 03 : run number
% 04 : trial number
% 05 : task
% 06 : fixation location number
% 07 : pursuit amplitude
% 08 : pursuit angle
% 09 : freeview imager number

end