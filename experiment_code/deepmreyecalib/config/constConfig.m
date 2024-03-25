function const = constConfig(scr, const)
% ----------------------------------------------------------------------
% const = constConfig(scr, const)
% ----------------------------------------------------------------------
% Goal of the function :
% Define all constant configurations
% ----------------------------------------------------------------------
% Input(s) :
% scr : struct containing screen configurations
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% ----------------------------------------------------------------------

% Randomization
[const.seed, const.whichGen] = ClockRandSeed;

% Colors
const.white = [255, 255, 255];
const.gray = [128 128 128];
const.black = [0,0,0];
const.fixation_color = const.white;
const.background_color = const.black; 

% Time parameters
const.TR_sec = 1.2;                                                         % MRI time repetition in seconds
const.TR_frm = round(const.TR_sec/scr.frame_duration);                      % MRI time repetition in seconds in screen frames

%new stimulus time parameters
const.iti_dur_TR = 5;                                                       % Inter trial interval duration in scanner TR
const.iti_dur_sec = const.iti_dur_TR * const.TR_sec;                        % Inter trial interval duration in seconds
const.iti_dur_frm = round(const.iti_dur_sec / scr.frame_duration);          % Inter trial interval in screen frames

const.fixtask_dur_TR = 1;                                                   % Fixation task stimulus duration in scanner TR
const.fixtask_dur_sec = const.fixtask_dur_TR * const.TR_sec;                % Fixation task stimulus duration in seconds
const.fixtask_dur_frm = round(const.fixtask_dur_sec / scr.frame_duration);  % Fixation task stimulus duration in screen frames

const.pursuit_dur_TR = 1;                                                   % Smooth pursuit task stimulus duration in scanner TR
const.pursuit_dur_sec = const.pursuit_dur_TR * const.TR_sec;                % Smooth pursuit task stimulus duration in seconds
const.pursuit_dur_frm = round(const.pursuit_dur_sec / scr.frame_duration);  % Smooth pursuit task stimulus duration in screen frames

const.freeview_dur_TR = 3;                                                  % Picture free viewing task stimulus duration in scanner TR
const.freeview_dur_sec = const.freeview_dur_TR * const.TR_sec;              % Picture free viewing task stimulus duration in seconds
const.freeview_dur_frm = round(const.freeview_dur_sec / scr.frame_duration);% Picture free viewing task stimulus duration in screen frames

% Stim parameters
[const.ppd] = vaDeg2pix(1, scr);                                            % one pixel per dva
const.dpp = 1/const.ppd;                                                    % degrees per pixel
const.window_sizeVal = 18;                                                  % side of the display window

% tasks
const.task_txt = {'inter-trial interval', 'fixation', 'pursuit', 'freeviewing'};

% fixation task
const.fixation_rows = 5;
const.fixation_cols = 5;
const.fixations_postions = const.fixation_rows * const.fixation_cols;
const.window_size = vaDeg2pix(const.window_sizeVal, scr);
const.fixations_postions_txt = {'[-7.0; +7.0]', '[-3.5; +7.0]', '[   0; +7.0]', '[+3.5; +7.0]', '[+7.0; +7.0]', ...
                                '[-7.0; +3.5]', '[-3.5; +3.5]', '[   0; +3.5]', '[+3.5; +3.5]', '[+7.0; +3.5]', ...
                                '[-7.0;    0]', '[-3.5;    0]', '[   0;    0]', '[+3.5;    0]', '[+7.0;    0]', ...
                                '[-7.0; -3.5]', '[-3.5; -3.5]', '[   0; -3.5]', '[+3.5; -3.5]', '[+7.0; -3.5]', ...
                                '[-7.0; -7.0]', '[-3.5; -7.0]', '[   0; -7.0]', '[+3.5; -7.0]', '[+7.0; -7.0]'};
const.fixation_coord_x = linspace(scr.x_mid - const.window_size/2, ...
                                  scr.x_mid + const.window_size/2, ...
                              const.fixation_cols);
                              
const.fixation_coord_y = linspace(scr.y_mid - const.window_size/2, ...
                                  scr.y_mid + const.window_size/2, ...
                              const.fixation_cols);
                          
const.fixation_coords = [];
for fix_cols = 1:const.fixation_cols
    for fix_rows = 1:const.fixation_rows
        
        const.fixation_coords = [const.fixation_coords;...
                                 const.fixation_coord_x(fix_rows), ...
                                 const.fixation_coord_y(fix_cols)];
    end
end

% pursuit task
const.pursuit_ampVal = [3, 5, 7];

const.pursuit_amp = vaDeg2pix(const.pursuit_ampVal, scr);
const.pursuit_amps = length(const.pursuit_ampVal);
const.pursuit_amps_txt = {'3.0 dva', '5.0 dva', '7.0 dva'};

const.pursuit_angles_steps = 20;
const.pursuit_angles = [0:const.pursuit_angles_steps:359];
const.pursuit_angles_txt = {'0 deg', '20 deg', '40 deg', '60 deg', ...
                            '80 deg', '100 deg', '120 deg', '140 deg', ...
                            '160 deg', '180 deg', '200 deg', '220 deg', ...
                            '240 deg', '260 deg', '280 deg', '300 deg', ...
                            '320 deg', '340 deg'};


% freeview task
const.freeview_pics = 10;
if ismac; const.freeview_path2pics = './stim/images';
else; const.freeview_path2pics = '.\stim\images';
end
const.freeview_pics_txt = {'water_drops', 'coffee', 'hands', 'astronaut', ...
                           'flat_iron', 'road', 'landscape', 'black swan', ...
                           'dog', 'balloon'};

% get image paths
path2pics = dir(fullfile(const.freeview_path2pics, 'image*'));
const.path2pics = fullfile(path2pics(1).folder, {path2pics(:).name}');
for pic_num = 1:length(const.path2pics)
    const.free_view_pic(:,:,:,pic_num) = imread(const.path2pics{pic_num});
end
const.freeview_pic_size = const.window_size;
const.freeview_pic_rect_orig = [0, 0, ...
                                size(const.free_view_pic,1),...
                                size(const.free_view_pic,2)];
const.freeview_pic_rect_disp = [ scr.x_mid - const.freeview_pic_size/2, ...
                                 scr.y_mid - const.freeview_pic_size/2, ...
                                 scr.x_mid + const.freeview_pic_size/2, ...
                                 scr.y_mid + const.freeview_pic_size/2];

% Trial settings
const.nb_repeat_fixation = 2;
const.nb_trials_fixation = const.fixation_rows * const.fixation_cols ...
    * const.nb_repeat_fixation;
const.TRs_fixation = const.nb_trials_fixation * const.fixtask_dur_TR;

const.nb_repeat_pursuit = 1;
const.nb_trials_pursuit = length(const.pursuit_ampVal) * ...
    length(const.pursuit_angles) * const.nb_repeat_pursuit;
const.TRs_pursuit = const.nb_trials_pursuit * const.pursuit_dur_TR;

const.nb_repeat_freeview = 1;
const.nb_trials_freeview = const.freeview_pics;
const.TRs_freeview = const.nb_trials_freeview * const.freeview_dur_TR;

const.nb_trials_iti = 4; % 3 iti and final one
const.TRs_iti = const.nb_trials_iti * const.iti_dur_TR;

const.nb_trials = const.nb_trials_fixation + const.nb_trials_pursuit + ...
    const.nb_trials_freeview + const.nb_trials_iti;

% define total TR numbers and scan duration
if const.scanner
    const.TRs_total = const.TRs_fixation + const.TRs_pursuit + ...
                            const.TRs_freeview + const.TRs_iti;
    fprintf(1,'\n\tScanner parameters: %1.0f TRs of %1.2f seconds for a total of %s\n',...
        const.TRs_total, const.TR_sec, ...
        datestr(seconds((const.TRs_total*const.TR_sec...
        )),'MM:SS'));
end

% Bullseye configs
const.fix_out_rim_radVal = 0.25;                                            % radius of outer circle of fixation bull's eye in dva
const.fix_rim_radVal = 0.75*const.fix_out_rim_radVal;                       % radius of intermediate circle of fixation bull's eye in dva
const.fix_radVal = 0.25*const.fix_out_rim_radVal;                           % radius of inner circle of fixation bull's eye in dva
const.fix_out_rim_rad = vaDeg2pix(const.fix_out_rim_radVal, scr);           % radius of outer circle of fixation bull's eye in pixels
const.fix_rim_rad = vaDeg2pix(const.fix_rim_radVal, scr);                   % radius of intermediate circle of fixation bull's eye in pixels
const.fix_rad = vaDeg2pix(const.fix_radVal, scr);                           % radius of inner circle of fixation bull's eye in pixels

% Personalised eyelink calibrations
angle = 0:pi/3:5/3*pi;
 
% compute calibration target locations
const.calib_amp_ratio = 0.5;
[cx1, cy1] = pol2cart(angle, const.calib_amp_ratio);
[cx2, cy2] = pol2cart(angle + (pi / 6), const.calib_amp_ratio * 0.5);
cx = round(scr.x_mid + scr.x_mid * [0 cx1 cx2]);
cy = round(scr.y_mid + scr.x_mid * [0 cy1 cy2]);
 
% order for eyelink
const.calibCoord = round([cx(1), cy(1),...                                  % 1. center center
    cx(9), cy(9),...                                                        % 2. center up
    cx(13),cy(13),...                                                       % 3. center down
    cx(5), cy(5),...                                                        % 4. left center
    cx(2), cy(2),...                                                        % 5. right center
    cx(4), cy(4),...                                                        % 6. left up
    cx(3), cy(3),...                                                        % 7. right up
    cx(6), cy(6),...                                                        % 8. left down
    cx(7), cy(7),...                                                        % 9. right down
    cx(10), cy(10),...                                                      % 10. left up
    cx(8), cy(8),...                                                        % 11. right up
    cx(11), cy(11),...                                                      % 12. left down
    cx(12), cy(12)]);                                                       % 13. right down

% compute validation target locations (calibration targets smaller radius)
const.valid_amp_ratio = const.calib_amp_ratio * 0.8;
[vx1, vy1] = pol2cart(angle, const.valid_amp_ratio);
[vx2, vy2] = pol2cart(angle + pi /6, const.valid_amp_ratio * 0.5);
vx = round(scr.x_mid + scr.x_mid*[0 vx1 vx2]);
vy = round(scr.y_mid + scr.x_mid*[0 vy1 vy2]);

% order for eyelink
const.validCoord =round([vx(1), vy(1),...                                   % 1. center center
    vx(9), vy(9),...                                                        % 2. center up
    vx(13), vy(13),...                                                      % 3. center down
    vx(5), vy(5),...                                                        % 4. left center
    vx(2), vy(2),...                                                        % 5. right center
    vx(4), vy(4),...                                                        % 6. left up
    vx(3), vy(3),...                                                        % 7. right up
    vx(6), vy(6),...                                                        % 8. left down
    vx(7), vy(7),...                                                        % 9. right down
    vx(10), vy(10),...                                                      % 10. left up
    vx(8), vy(8),...                                                        % 11. right up
    vx(11), vy(11),...                                                      % 12. left down
    vx(12), vy(12)]);                                                       % 13. right down
end