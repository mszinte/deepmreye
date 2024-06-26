function expDes = runTrials(scr, const, expDes, my_key)
% ----------------------------------------------------------------------
% expDes = runTrials(scr, const, expDes, my_key)
% ----------------------------------------------------------------------
% Goal of the function :
% Draw stimuli of each indivual trial and waiting for inputs
% ----------------------------------------------------------------------
% Input(s) :
% scr : struct containing screen configurations
% const : struct containing constant configurations
% expDes : struct containg experimental design
% my_key : structure containing keyboard configurations
% ----------------------------------------------------------------------
% Output(s):
% resMat : experimental results (see below)
% expDes : struct containing all the variable design configurations.
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

trial_pursuit = 0;
for t = 1:const.nb_trials

    % Open video
    if const.mkVideo
        open(const.vid_obj);
    end
    
    % Compute and simplify var and rand
    task = expDes.expMat(t, 5);
    var1 = expDes.expMat(t, 6);
    var2 = expDes.expMat(t, 7);
    var3 = expDes.expMat(t, 8);
    var4 = expDes.expMat(t, 9);

    % Check trial
    if const.checkTrial && const.expStart == 0
        fprintf(1,'\n\n\t=================== TRIAL %3.0f ====================\n',t);
        fprintf(1,'\n\tTask =             \t%s', const.task_txt{task});
        if ~isnan(var1); fprintf(1,'\n\tFixation location =\t%s', ...
                const.fixations_postions_txt{var1}); end
        if ~isnan(var2); fprintf(1,'\n\tPursuit amplitude =\t%s', ...
                const.pursuit_amps_txt{var2}); end
        if ~isnan(var3); fprintf(1,'\n\tPursuit angle =    \t%s', ...
                const.pursuit_angles_txt{var3}); end
        if ~isnan(var4); fprintf(1,'\n\tPicture =          \t%s', ...
                const.freeview_pics_txt{var4}); end
    end
    
    % Timing
    switch task
        case 1
            iti_onset_nbf = 1;
            iti_offset_nbf = const.iti_dur_frm;
            trial_offset = iti_offset_nbf;
        case 2
            fix_onset_nbf = 1;
            fix_offset_nbf = const.fixtask_dur_frm;
            trial_offset = fix_offset_nbf;
        case 3
            trial_pursuit = trial_pursuit + 1;
            pursuit_onset_nbf = 1;
            pursuit_offset_nbf = const.pursuit_dur_frm;
            trial_offset = pursuit_offset_nbf;
        case 4
            
            freeview_onset_nbf = 1;
            freeview_offset_nbf = const.freeview_dur_frm;
            trial_offset = freeview_offset_nbf;
    end
    
    % Compute fixation coordinates
    if task == 1
        iti_x = scr.x_mid;
        iti_y = scr.y_mid;
    end
    
    % Compute fixation coordinates
    if task == 2
        fix_x = const.fixation_coords(var1, 1);
        fix_y = const.fixation_coords(var1, 2);
    end

    % Compute pursuit coordinates
    if task == 3
        pursuit_amp = const.pursuit_amp(var2);
        pursuit_angle = const.pursuit_angles(var3);
        
        if trial_pursuit == 1
            pursuit_coord_on = [scr.x_mid, scr.y_mid];
            pursuit_coord_off = pursuit_coord_on + [pursuit_amp * cosd(pursuit_angle), ...
                                                    pursuit_amp * -sind(pursuit_angle)];
        elseif trial_pursuit == const.nb_trials_pursuit
            pursuit_coord_on = pursuit_coord_off;
            pursuit_coord_off = [scr.x_mid, scr.y_mid];
        else
            pursuit_coord_on = pursuit_coord_off;
            pursuit_coord_off = pursuit_coord_on + [pursuit_amp * cosd(pursuit_angle), ...
                                                    pursuit_amp * -sind(pursuit_angle)];
        end

        purs_x = linspace(pursuit_coord_on(1), pursuit_coord_off(1), const.pursuit_dur_frm);
        purs_y = linspace(pursuit_coord_on(2), pursuit_coord_off(2), const.pursuit_dur_frm);

    end
    
    % Get freeview image
    if task == 4
        pic_tex = Screen('MakeTexture', scr.main, const.free_view_pic(:,:,:,var4));
    end

    % Wait first MRI trigger
    if t == 1
        Screen('FillRect',scr.main,const.background_color);
        drawBullsEye(scr, const, scr.x_mid, scr.y_mid, 0);
        Screen('Flip',scr.main);
    
        first_trigger = 0;
        expDes.mri_band_val = my_key.first_val(end);
        while ~first_trigger
            if const.scanner == 0 || const.scannerTest
                first_trigger = 1;
                mri_band_val = -8;
            else
                keyPressed = 0;
                keyCode = zeros(1,my_key.keyCodeNum);
                for keyb = 1:size(my_key.keyboard_idx, 2)
                    [keyP, keyC] = KbQueueCheck(my_key.keyboard_idx(keyb));
                    keyPressed = keyPressed + keyP;
                    keyCode = keyCode + keyC;
                end
                if const.scanner == 1
                    input_return = [my_key.ni_session2.inputSingleScan,...
                        my_key.ni_session1.inputSingleScan];
                    if input_return(my_key.idx_mri_bands) == ...
                            ~expDes.mri_band_val
                        keyPressed = 1;
                        keyCode(my_key.mri_tr) = 1;
                        expDes.mri_band_val = ~expDes.mri_band_val;
                        mri_band_val = input_return(my_key.idx_mri_bands);
                    end
                end
                if keyPressed
                    if keyCode(my_key.escape) && const.expStart == 0
                        overDone(const, my_key)
                    elseif keyCode(my_key.mri_tr)
                        first_trigger = 1;
                        mri_band_val = -8;
                    end
                end
            end
        end
        
       % Write in edf file
        log_txt = sprintf('trial %i mri_trigger val = %i', t, ...
            mri_band_val);
        if const.tracker; Eyelink('message', '%s', log_txt); end
    end
    
    
    % Write in edf file
    if const.tracker
        Eyelink('message', '%s', sprintf('trial %i started\n', t));
    end
    
    % Main diplay loop
    nbf = 0;
    while nbf < trial_offset
        % Flip count
        nbf = nbf + 1;
    
        Screen('FillRect', scr.main, const.background_color)
    
        % Inter-trial interval
        if task == 1
            if nbf >= iti_onset_nbf && nbf <= iti_offset_nbf 
                drawBullsEye(scr, const, iti_x, iti_y, 1);
            end
        end
        
        % Fixation task
        if task == 2
            if nbf >= fix_onset_nbf && nbf <= fix_offset_nbf
                drawBullsEye(scr, const, fix_x, fix_y, 1);
            end
        end
        
        % Pursuit task
        if task == 3
            if nbf >= pursuit_onset_nbf && nbf <= pursuit_offset_nbf
                drawBullsEye(scr, const, purs_x(nbf), purs_y(nbf), 1);
            end
        end
        
        % Freeview task
        if task == 4
            if nbf >= freeview_onset_nbf && nbf <= freeview_offset_nbf
                Screen('DrawTexture', scr.main, pic_tex, ...
                    const.freeview_pic_rect_orig, ...
                    const.freeview_pic_rect_disp);
            end
            if nbf == freeview_offset_nbf
                Screen('Close', pic_tex);
            end            
        end
        
        % Check keyboard
        keyPressed = 0;
        keyCode = zeros(1,my_key.keyCodeNum);
        for keyb = 1:size(my_key.keyboard_idx,2)
            [keyP, keyC] = KbQueueCheck(my_key.keyboard_idx(keyb));
            keyPressed = keyPressed + keyP;
            keyCode = keyCode + keyC;
        end
        if keyPressed
            if keyCode(my_key.escape) && const.expStart == 0
                overDone(const, my_key)
            end
        end
        
        % flip screen
        vbl = Screen('Flip', scr.main);
        
        % Create movie
        if const.mkVideo
            expDes.vid_num = expDes.vid_num + 1;
            image_vid = Screen('GetImage', scr.main);
            imwrite(image_vid,sprintf('%s_frame_%i.png', ...
                const.movie_image_file, expDes.vid_num))
            writeVideo(const.vid_obj,image_vid);
        end
        
        % Save trials times
        if task == 1
            if nbf == iti_onset_nbf
                trial_on = vbl;
                log_txt = sprintf('iti %i onset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            elseif nbf == iti_offset_nbf
                log_txt = sprintf('iti %i offset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            end
        elseif task == 2
            if nbf == fix_onset_nbf
                trial_on = vbl;
                log_txt = sprintf('fixation %i onset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            elseif nbf == fix_offset_nbf
                log_txt = sprintf('fixation %i offset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            end
        elseif task == 3
            if nbf == pursuit_onset_nbf
                trial_on = vbl;
                log_txt = sprintf('pursuit %i onset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            elseif nbf == pursuit_offset_nbf
                log_txt = sprintf('pursuit %i offset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            end
        elseif task == 4
            if nbf == freeview_onset_nbf
                trial_on = vbl;
                log_txt = sprintf('freeview %i onset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            elseif nbf == freeview_offset_nbf
                log_txt = sprintf('freeview %i offset at %f', t, vbl);
                if const.tracker; Eyelink('message','%s',log_txt); end
            end
        end
    end
    expDes.expMat(t, 1) = trial_on;
    expDes.expMat(t, 2) = vbl - trial_on;

    % Write in log/edf
    if const.tracker
        Eyelink('message', '%s', sprintf('trial %i ended\n', t));
    end


end

end

