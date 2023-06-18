clc;clear;

h = 896;
w = 1280;

fid = fopen('result.bin', 'rb');
data = fread(fid, 'double');
data = reshape(data, [w, h]);
data = data';
fclose(fid);