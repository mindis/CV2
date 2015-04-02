function [ transformedImage ] = transformImage( im1, x, returnZeroAsNaN)
%% transformImage
% im1 = basis image
% x = transformation matrix [m1,m2,m3,m4,t1,t2]

    % create default image matrix (for testing purposes)
    if (nargin < 1) 
        im1 = imread('left.jpg');
    end
        
    % create default transformation matrix (for testing purposes)
    if (nargin < 2) 
        x = [2.34295628439813; %scale x
             0.322216686754101; % rotation x
             -0.300258849013665; % rotation y
             2.44730444413627; %scale y
             -680.668593715442;
             -368.307845509866];
    end
   
    im1 = im2double(im1);
    
    %# Set new image size [rows cols]   
    oldSize = size(im1);
    oldCorners = [1 ,1          ,oldSize(1) ,oldSize(1); 
                  1 ,oldSize(2) ,1          ,oldSize(2);
                  1 ,1          ,1          ,1];

    % scale and predict newImage size
    M = [ x(1) x(2) 0;
          x(3) x(4) 0;
          0    0    1]';
      
    product = M * oldCorners;          
   
    height = floor(max(product(1,:)) + abs(x(6)) );
    width = floor(max(product(2,:))+ abs(x(5)) );
    [colIndex, rowIndex] = meshgrid(1:height, 1:width);
   
    M_ = [ 
        x(1) x(2) 0;
        x(3) x(4) 0;
        x(6) x(5) 1
        ]';

    imTraceback = floor(M_ \ [colIndex(:), rowIndex(:), ones(length(rowIndex(:)),1)]');
    
    rowIndex = reshape(imTraceback(1,:),width,height)';
    colIndex = reshape(imTraceback(2,:),width,height)';
    
    %normalizing rowIndex and colIndex
    rowIndex = rowIndex;
    colIndex = colIndex;% + min(imTraceback(1,:));
    
    if (numel(oldSize)==2)
        transformedImage = NaN(height, width);
    else 
        transformedImage = NaN(height, width, oldSize(3));
    end
    
    for col = 1:width
        for row = 1:height
            rIndex = rowIndex(row, col);
            cIndex = colIndex(row, col);
            if ~(rIndex < 1 || rIndex > oldSize(1) || cIndex < 1 || cIndex > oldSize(2))
                transformedImage( row, col, :) = im1(rIndex, cIndex, :);
            end
            
        end
    end
       
    if (nargin < 3)
        transformedImage(isnan(transformedImage)) = 0 ;        
    end
    
end

