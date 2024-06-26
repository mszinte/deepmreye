function drawTrialInfoEL(const)
% ----------------------------------------------------------------------
% drawTrialInfoEL(const)
% ----------------------------------------------------------------------
% Goal of the function :
% Draw on the eyelink display the experiment configuration
% ----------------------------------------------------------------------
% Input(s) :
% const : struct containing constant configurations
% ----------------------------------------------------------------------
% Output(s):
% none
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Edited by Sina KLING (sina.kling@outlook.de)
% ----------------------------------------------------------------------

% o--------------------------------------------------------------------o
% | EL Color index                                                     |
% o----o----------------------------o----------------------------------o
% | Nb |  Other(cross,box,line)     | Clear screen                     |
% o----o----------------------------o----------------------------------o
% |  0 | black                      | black                            |
% o----o----------------------------o----------------------------------o
% |  1 | dark blue                  | dark dark blue                   |
% o----o----------------------------o----------------------------------o
% |  2 | dark green                 | dark blue                        |
% o----o----------------------------o----------------------------------o
% |  3 | dark turquoise             | blue                             |
% o----o----------------------------o----------------------------------o
% |  4 | dark red                   | light blue                       |
% o----o----------------------------o----------------------------------o
% |  5 | dark purple                | light light blue                 |
% o----o----------------------------o----------------------------------o
% |  6 | dark yellow (brown)        | turquoise                        |
% o----o----------------------------o----------------------------------o
% |  7 | light gray                 | light turquoise                  | 
% o----o----------------------------o----------------------------------o
% |  8 | dark gray                  | flashy blue                      |
% o----o----------------------------o----------------------------------o
% |  9 | light purple               | green                            |
% o----o----------------------------o----------------------------------o
% | 10 | light green                | dark dark green                  |
% o----o----------------------------o----------------------------------o
% | 11 | light turquoise            | dark green                       |
% o----o----------------------------o----------------------------------o
% | 12 | light red (orange)         | green                            |
% o----o----------------------------o----------------------------------o
% | 13 | pink                       | light green                      |
% o----o----------------------------o----------------------------------o
% | 14 | light yellow               | light green                      |
% o----o----------------------------o----------------------------------o
% | 15 | white                      | flashy green                     |
% o----o----------------------------o----------------------------------o

% Color config
frameCol = 15;
ftCol = 15;
bgCol = 0;

% Clear screen
eyeLinkClearScreen(bgCol);

% Draw Stimulus
for tpos = 1:size(const.fix_coords,1)
    eyeLinkDrawBox(const.fix_coords(tpos,1), const.fix_coords(tpos,2), ...
        const.fix_out_rim_rad * 2, const.fix_out_rim_rad * 2, 2, frameCol, ftCol);
end

WaitSecs(0.1);

end