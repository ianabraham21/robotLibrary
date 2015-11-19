%% Load the Robotics Toolbox into Matlab's path
run('robot-9.10/rvctools/startup_rvc.m') % you only have to run this once

%% Load the WAM model into the workspace
% load the 7-dof Barrett WAM model into the workspace.
mdl_wam

% mdl_wam creates a variable "wam", which represents a WAM arm
wam % output the arm to the command window
% wam is a SerialLink object, which has various functions and properties

% learn more about the variables and functions of a SerialLink object
% doc SerialLink -or- help SerialLink

%% Animate a trajectory using the model
n = wam.n; % number of joints
% q0 = zeros(1, n); % home configuration
% qf = rand(1, n)*2*pi; % final configuration
% % qf = [ -5.54210275   6.61456082  -7.29589464 -16.67909455   0.38380961 10.25283076  13.07931786];
% qf = [  8.5591218    5.84737394 -20.71232978 -16.67846922  -3.67848633 14.79613082  14.97338326]
% qf = sign(qf).*mod(abs(qf),2*pi);
% k = 50; % create k points on a path from q0 to qf
% 
% % generate a linear path between q0 and qf
% Q = zeros(k, n);
% for i = 1:n
%     Q(:, i) = linspace(q0(i), qf(i), k);
% end


Q = csvread('WAM_test.csv');


% Q is now a k x n matrix representing a path in configuration space of
% the manipulator.  Each row in the matrix is a configuration along the
% path, where Q(1,:) is the initial configuration q0 and Q(k,:) is the
% final configuration qf.

% an alternative way to generate a path using jtraj from the toolbox
% Q = jtraj(q0, qf, k); % help jtraj to learn more

% animate the trajectory of the robot arm --remember each row is a new pose
wam.plot(Q) % all SerialLink objects, like wam, have a plot function

%% Plot a single pose
% call plot with a 1 x n row vector for a single configuration
% wam.plot(q0)
% wam.plot(Q(1,:))

%% Done!
% see robot.pdf for more info about the Robotics Toolbox.
% open('robot.pdf')

%% The UR5 manipulator
% The other model we'll be working with is the Universal UR5 arm.
% Uncomment the last two lines in this file to load the UR5 model into the
% workspace or create a new file and, using the lines above as reference,
% replace ur5 with wam to get a plot for the Universal UR5 manipulator.

% mdl_ur5;
% ur5  % output the arm to the command window