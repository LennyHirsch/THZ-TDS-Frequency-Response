parsons = readmatrix('parsons.csv');
wvl = parsons(:,1).*10^-6;
n = parsons(:,2);


plot(wvl, n)