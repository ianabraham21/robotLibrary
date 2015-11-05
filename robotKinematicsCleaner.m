data = csvread('provideArgumentName.csv')

newdata = [data(:,1),mod(abs(data(:,2:end)),pi).*sign(data(:,2:end))

csvwrite('fixedValues.csv',newdata)
