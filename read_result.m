clc;clear;

h = 896;
w = 1280;

fid = fopen('data/5000_disappear_origin.bin', 'rb');
data = fread(fid, 'float');
data = reshape(data, [w, h]);
data = data';
fclose(fid);

data2 = data / max(data(:));
data2(data2 < 0.3) = 0;
data2(data2 > 0.7) = 0;



