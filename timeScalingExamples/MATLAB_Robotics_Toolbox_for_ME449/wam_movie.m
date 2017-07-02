%% Load the Robotics Toolbox into Matlab's path
run('robot-9.10/rvctools/startup_rvc.m') % you only have to run this once

%% Load the WAM model into the workspace
mdl_wam

%% Generate a trajectory using the model
n = wam.n;
q0 = zeros(1, n); % home configuration
qf = rand(1, n)*2*pi; % final configuration
k = 25; % create k points on a path from q0 to qf
Q = jtraj(q0, qf, k); % jtraj is part of RTB; help jtraj for more info

%% Save each configuration as an image
wam.plot(Q, 'movie', 'wam_imgs') % save poses as PNGs in folder wam_imgs

%% Make a movie
% following code taken from
% http://www.mathworks.com/help/matlab/examples/convert-between-image-sequences-and-video.html

% get PNGs from wam_imgs
imageNames = dir(fullfile('wam_imgs','*.png'));
imageNames = {imageNames.name}';

% write to video file
fps = 10; % frames per second; this is the default value in plot

% option 1: create an AVI file
outputVideo = VideoWriter(fullfile('wam_out.avi'));
% option 2: create an MP4 file
% outputVideo = VideoWriter(fullfile('wam_out.mp4'));
outputVideo.FrameRate = fps;
open(outputVideo)

for ii = 1:length(imageNames)
   img = imread(fullfile('wam_imgs',imageNames{ii}));
   writeVideo(outputVideo,img)
end

close(outputVideo)

% the video is now in your current folder, play the video using your
% favorite video player