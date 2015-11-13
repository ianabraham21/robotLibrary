
% ur5
%theta = [ -3.26139825e+00  -2.56717457e+03   5.34838526e+03  -2.73094744e+03  -3.26125144e+00  -4.55553176e+01];
% WAM
theta = [  6.21695041   7.63359732  -0.10899343  -7.99376428 -19.13988724   6.65937501   6.64404185];


theta = [10,mod(abs(theta),2*pi).*sign(theta)];
theta0 = zeros(size(theta));

data = linspace(theta0,theta,25)'

%characterPara = {'time','shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'};

%time,base_yaw_joint,shoulder_pitch_joint,shoulder_yaw_joint,elbow_pitch_joint,wrist_yaw_joint,wrist_pitch_joint,palm_yaw_joint

csvwrite('WAM_test.csv',data)
