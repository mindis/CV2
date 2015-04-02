im1 = imread('img1.pgm');
im2 = imread('img5.pgm');

x = ransac(im1, im2, 4, 5);

I1 = transformImage( im1, x );
imshowpair(I1,im2)

% transform image 1 to image 2