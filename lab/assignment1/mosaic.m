function [outputImg] = mosaic(im1, im2)
    % switch the images, so we can transform the im2 instead of im1
    x = ransac(im2, im1, 4, 5);

    % transform im2
    x_ = x;
    x_(5) = max(0,x(5));
    x_(6) = max(0,x(6));
    im2_ = transformImage( im2, x_, 'returnZeroAsNaN' );

    %im1 is basis image, so we will not scale/rotate it.
    x_ = [1; 0; 0; 1; 0; 0];
    x_(5) = abs(min(0,x(5)));
    x_(6) = abs(min(0,x(6)));

    im1_ = transformImage( im1, x_, 'returnZeroAsNaN' );

    % normalize size
    % -> row
    enlargeRow = abs(size(im2_,1) - size(im1_,1));
    if (size(im2_,1) < size(im1_,1))
        im2_ = vertcat(im2_, NaN(enlargeRow, size(im2_,2), size(im2_,3)));
    else
        im1_ = vertcat(im1_, NaN(enlargeRow, size(im1_,2), size(im1_,3)));
    end

    % -> col
    enlargeCol = abs(size(im2_,2) - size(im1_,2));
    if (size(im2_,2) < size(im1_,2))
        im2_ = horzcat(im2_, NaN(size(im2_,1), enlargeCol, size(im2_,3)));
    else
        im1_ = horzcat(im1_, NaN(size(im1_,1), enlargeCol, size(im1_,3)));
    end

    % combine + normalize(blend)
    im1_(~isnan(im2_)) = NaN;  
    
    im1_(isnan(im1_)) = 0 ;
    im2_(isnan(im2_)) = 0 ;
    outputImg = im1_ + im2_;
end