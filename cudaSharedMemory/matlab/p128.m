%% SCRIPT: PIPELINE_NON_LOCAL_MEANS
%
% Pipeline for non local means algorithm as described in [1].
%
% The code thus far is implemented in CPU.
%
% DEPENDENCIES
%
% [1] Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local
%     algorithm for image denoising. In 2005 IEEE Computer Society
%     Conference on Computer Vision and Pattern Recognition (CVPR’05),
%      volume 2, pages 60–65. IEEE, 2005.
%
  
  clear all %#ok
  close all

  %% PARAMETERS
  
  % input image
  pathImg   = '../data/rossi_128.mat';
  strImgVar = 'rossi_128';
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  filtSigma = 0.02;
  patchSize = [7 7];
  patchSigma = 5/3;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));
  
  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  ioImg = matfile( pathImg );
  I     = ioImg.(strImgVar);
  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  figure('Name','Original Image');
  imagesc(I); axis image;
  colormap gray;
  savefig('results/original128.fig');
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  savefig('results/noisy128.fig');
  %% NON LOCAL MEANS
  
  %tic;
  %If = nonLocalMeans( J, patchSize, filtSigma, patchSigma );
  %toc
    
  %% NON LOCAL MEANS CUDA
  
  %tic;
  f = sample_kernel( J, patchSize, filtSigma, patchSigma );
  %toc
 %imwrite(f,'output.jpg');
  %% VISUALIZE RESULT
  
  %figure('Name', 'Filtered image');
  %imagesc(If); axis image;
  %colormap gray;
  %savefig('filtered.fig');
  figure('Name', 'Filtered Cuda image');
  imagesc(f); axis image;
  colormap gray;
  savefig('results/output128.fig');
  %figure('Name', 'Residual');
  %imagesc(If-J); axis image;
  %colormap gray;
  %savefig('res.fig');
  %% (END)
 CudaPsnr = psnr(f,single(I),1) 
  fprintf('...end %s...\n',mfilename);


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 28, 2016
%
% CHANGELOG
%
%   0.1 (Dec 28, 2016) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------
