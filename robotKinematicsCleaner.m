data = csvread('newdata.csv')

characterPara = ["time","shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"];
newdata = [characterPara;data(:,1),mod(abs(data(:,2:end)),pi).*sign(data(:,2:end))]


csvwrite('scriptTest_temp.csv',newdata)