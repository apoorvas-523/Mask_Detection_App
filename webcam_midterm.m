cam = webcam;
 preview(cam);
load('Trained_network.mat');
while 1
img = snapshot(cam);
% The trained network has 80x80x3
% The size of the image obtained in the web cam is 1280x720x3
% img = rgb2gray(img);
% 280 columns from each side
% img(:,1:280)=[];
% img(:,end-279:end)=[];
img=imresize(img,[80 80]);
output=classify(net,img);
figure(1)
imagesc(img);
title(output);
end
clear cam