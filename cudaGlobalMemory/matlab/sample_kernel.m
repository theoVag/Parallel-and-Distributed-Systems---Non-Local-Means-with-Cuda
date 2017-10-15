function f = sample_kernel(I, patchSize, filtSigma, patchSigma)
%% SCRIPT: SAMPLE_KERNEL
%
% Sample usage of GPU kernel through MATLAB
%
% DEPENDENCIES
%
%  sampleAddKernel.cu
%
tic  
 %% clear variables;

  %% PARAMETERS
  
  threadsPerBlock = [32 32];
  %m = 1000;
  %n = 1000;
  % [m,n]=size(I);
	
	%patchSize=patchSize(1,1);
  %% (BEGIN)
	[m n] =size(I);
	I=padarray(I,(patchSize-1)/2,'symmetric','both');
	
  fprintf('...begin %s...\n',mfilename);  
  
  %% KERNEL
  
  z = parallel.gpu.CUDAKernel( '../cuda/sampleAddKernel.ptx', ...
                               '../cuda/sampleAddKernel.cu','Zev');
  fc = parallel.gpu.CUDAKernel( '../cuda/sampleAddKernel.ptx', ...
                               '../cuda/sampleAddKernel.cu','fev');

  numberOfBlocks  = ceil( [m n] ./ threadsPerBlock );
  
 % b.ThreadBlockSize = threadsPerBlock;
 % b.GridSize        = numberOfBlocks;
  z.ThreadBlockSize = threadsPerBlock;
  z.GridSize        = numberOfBlocks;
  fc.ThreadBlockSize = threadsPerBlock;
  fc.GridSize        = numberOfBlocks;
  %% DATA
    %bazw single gia na ta kanw float : matrix = zeros(10, 8, 'single');matrix = single(rand(10,8));
  I=single(I);
  Ad = gpuArray(I);
  [m n] =size(I);

  Zd = zeros([n m], 'gpuArray');
  f = zeros([n m], 'gpuArray');

Zd=single(Zd);
f=single(f);
  
  %Ad=reshape(
  %Ad=reshape(Ad,1,n*m);
  %Bd=reshape(Bd,1,n*m);
  %Zd=reshape(Zd,1,n*m);
  %fd=reshape(fd,1,n*m);
  %wd=reshape(wd,1,m*n*n*m);
  %Bd=reshape(1:m*n,n,m);
  %Zd=reshape(1:m*n,n,m);
  %fd=reshape(1:m*n,n,m);
  % gaussian patch
  H = fspecial('gaussian',patchSize, patchSigma);
  H = H(:) ./ max(H(:));
  H=single(H);
  

  Zd = gather(feval(z,Ad,Zd,H, m, n,patchSize(1,1),patchSigma,filtSigma));
  f=   gather(feval(fc, Ad,Zd, f, H, m, n,patchSize(1,1),patchSigma,filtSigma));
  
  %f = gather(feval(fc, Ad,Zd,Bd, fd, H, m, n,patchSize(1,1),patchSigma,filtSigma) );

 f=f((1+(patchSize(1,1)-1)/2):(m-(patchSize(1,1)-1)/2),(1+(patchSize(1,1)-1)/2):(m-(patchSize(1,1)-1)/2)); 
  
  %% SANITY CHECK
  
  %fprintf('Error: %e\n', norm( Bd - (Ad+1), 'fro' ) );
  
  %% (END)
toc
  fprintf('...end %s...\n',mfilename);
 %f = reshape( f, [n m] );%mporei na xreiazetai
end
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

