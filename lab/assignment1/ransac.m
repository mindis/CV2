function [x, score] = ransac(im1, im2, p, n)

    % make single
    im1_ = im2single(im1) ;
    im2_ = im2single(im2) ;

    % make grayscale
    if size(im1_,3) > 1, im1g = rgb2gray(im1_) ; else im1g = im1_ ; end
    if size(im2_,3) > 1, im2g = rgb2gray(im2_) ; else im2g = im2_ ; end

    % extract sift frame and descriptor
    [f1,d1] = vl_sift(im1g) ;
    [f2,d2] = vl_sift(im2g) ;

    % find matches
    [matches, scores] = vl_ubcmatch(d1,d2) ;

    % helpers variables
    numMatches = size(matches,2) ;
    X1 = f1(1:2,matches(1,:));
    X2 = f2(1:2,matches(2,:));

    if (nargin < 4)
        n = 100;
    end
    
    if (nargin < 3)
        p = 4;
    end
    
    x = [];
    score = zeros(n, 1);
    for t = 1:n
      % estimate transform parameter
      subset = vl_colsubset(1:numMatches, p) ;
      A = [];
      b = [];
      for i = subset   
          A = cat(1, A, [ X1(1,i), X1(2,i),       0,       0, 1, 0;
                                0,       0, X1(1,i), X1(2,i), 0, 1] );

          b = cat(1, b, X2(:,i));    
      end

      x = cat(2, x, pinv(A)*b);

      % score each candidate
      A = [];
      b = [];
      for i = 1:numMatches   
          A = cat(1, A, [ X1(1,i), X1(2,i),       0,       0, 1, 0;
                                0,       0, X1(1,i), X1(2,i), 0, 1] );

          b = cat(1, b, X2(:,i));    
      end

      bPred = A*x(:,t);
      bPred = (bPred - b).^2;
      bPred = reshape(bPred, [2, numMatches]);

      score(t) = sum((bPred(1,:) + bPred(1,:)) < 10*10) ;
    end

    % pick the best candidate
    [score, best] = max(score) ;
    x = x(:,best);

end