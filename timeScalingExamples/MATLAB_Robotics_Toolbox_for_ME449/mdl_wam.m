%MDL_WAM Create model of Barrett WAM manipulator
%
% MDL_WAM is a script that creates the workspace variable wam which
% describes the kinematic and dynamic characteristics of a 7-DOF
% Barrett WAM manipulator using standard DH conventions. 
%
%
% Notes::
% - SI units are used.
%
% Reference::
% - "http://support.barrett.com/wiki/WAM/KinematicsJointRangesConversionFactors",
%   Last accessed Oct. 2015.
%
% See also SerialRevolute, mdl_puma560akb, mdl_stanford.

% MODEL: Barrett, WAM, standard_DH

clear L

%##########################################################
% By default, all joints are revolute joints.  The model uses the standard
% DH convention.  See RTB documentation for more details.

%            theta    d      a      alpha
L(1) = Link([ 0       0      0      -pi/2]);
L(2) = Link([ 0       0      0       pi/2]);
L(3) = Link([ 0       0.55   0.045  -pi/2]);
L(4) = Link([ 0       0     -0.045   pi/2]);
L(5) = Link([ 0       0.3    0      -pi/2]);
L(6) = Link([ 0       0      0       pi/2]);
L(7) = Link([ 0       0.06   0       0]);

%##########################################################

T = transl(0,0,0.346); % add offset to line up with rviz/URDF base coords.
wam = SerialLink(L, 'name', 'WAM', 'base', T);
wam.model3d = 'BARRETT/wam';
wam.plotopt = {'workspace', [-1 1 -1 1 -1 1.7], 'scale', 0.6};

%##########################################################

clear L T