clc;clear;
root = 'data/500_ori';
out = 'output';
mkdir(out);
files = dir(root);
for t=3:length(files)
    name = files(t).name;
    if contains(name, '.txt')
        continue
    end
    
    disp(name);
    path = [root '/' name];
    img = imread(path);
    img = rgb2gray(img);
  
    img = double(img)/255;
    gray_ori = img;
    img = imresize(img, 0.5);
    
    [h,w] = size(img);
    h2 = h/2;
    w2 = w/2;
    r = min(h2,w2)*0.9;
    
    for i=1:h
        for j=1:w
            ii = i - h2;
            jj = j - w2;
            if ii*ii + jj*jj >= r*r
                img(i,j) = 0;
            else
                g = img(i,j);
                g = g*g;
                if g > 0.01
                    img(i,j) = 1;
                else
                    img(i,j) = 0;
                end
            end
        end
    end
    
    sum(img(:))
    
    img = medfilt2(img,[5 5]);
    disk1=strel('disk',1);
    disk2=strel('disk',2);
    disk3=strel('disk',3);
    img=imdilate(img,disk1);
    img=imerode(img,disk3);
    img=imdilate(img,disk2);

    img = imresize(img, size(gray_ori));
    imwrite(uint8(img*255), [out, '/' name(1:end-4) '_seg.jpg']);
    
%     gray = gray_ori.*gray;
%     break; 
end