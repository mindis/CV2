tic
im1 = imread('left.jpg');
im2 = imread('right.jpg');

imshow(mosaic(im1, im2));
toc

% %testing combining ships images
% for i = 2:6
%     if (i==2)
%         im1 = imread('img1.pgm');
%     else
%         im1 = combined;
%     end
%     
%     im2 = imread(sprintf('img%d.pgm',i));
% 
%     combined = mosaic(im1, im2);
%     
%     toc   
% end
% figure, imshow(combined);