function f = sample_kernel(I, patchSize, filtSigma, patchSigma)
%% SCRIPT: SAMPLE_KERNEL
%
% Sample usage of GPU kernel through MATLAB
%
% DEPENDENCIES
%
%  sampleAddKernel.cu
%
  
  %clear variables;
tic
  %% PARAMETERS
  [k v]=size(I); %size before padding
  threadsPerBlock = [32 32];
  [m n]=size(I); % same just for use in numberOfBlocks
  I=padarray(I,(patchSize-1)/2,'symmetric','both');

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  
  
  %% KERNEL
  
 % k = parallel.gpu.CUDAKernel( '../cuda/sampleAddKernel.ptx', ...
  %                             '../cuda/sampleAddKernel.cu');
    z = parallel.gpu.CUDAKernel( '../cuda/sampleAddKernel.ptx', ...
                               '../cuda/sampleAddKernel.cu','Zev');
    t = parallel.gpu.CUDAKernel( '../cuda/sampleAddKernel.ptx', ...
                               '../cuda/sampleAddKernel.cu','fev');

  numberOfBlocks  = ceil( [m n] ./ threadsPerBlock );
  z.SharedMemorySize = 16000;
  z.ThreadBlockSize = threadsPerBlock;
  z.GridSize        = numberOfBlocks;
  t.SharedMemorySize = 16000;
  t.ThreadBlockSize = threadsPerBlock;
  t.GridSize        = numberOfBlocks;
  %% DATA
    I=single(I);
  Ad = gpuArray(I); %move image to gpu
  [m n] =size(I); % m,n size after padding

  Zd = zeros([m n], 'gpuArray'); %fill with zeros array in gpu for returning Z
  f = zeros([m n], 'gpuArray'); %fill with zeros array in gpu for returning f (filteredImage)
  Zd=single(Zd); % convert arrays to float
  f=single(f);
  %calculate gaussian for our functions
  H = fspecial('gaussian',patchSize, patchSigma);
  H = H(:) ./ max(H(:));
  H=single(H);

  w=m-(patchSize(1,1)-1);
  h=n-(patchSize(1,2)-1);

  for j=[1 threadsPerBlock(1,1):threadsPerBlock(1,1):w-threadsPerBlock(1,1)]
	
     for i= [1 threadsPerBlock(1,2):threadsPerBlock(1,2):h-threadsPerBlock(1,2)]
        x=i:i+threadsPerBlock(1,1)+patchSize(1,1)-2;
        y=j:j+threadsPerBlock(1,2)+patchSize(1,2)-2;
	
        Zd = (feval(z,Ad,Ad(x,y),Zd,H, m, n,patchSize(1,1),filtSigma));
	wait(gpuDevice);
     end
  end
  Zd=gather(Zd);

%----------------------------------------------------------------------------------------------

    w=m-(patchSize(1,1)-1);
  h=n-(patchSize(1,2)-1);

  for j=[1 threadsPerBlock(1,1):threadsPerBlock(1,1):w-threadsPerBlock(1,1)]
	
     for i= [1 threadsPerBlock(1,2):threadsPerBlock(1,2):h-threadsPerBlock(1,2)]
        x=i:i+threadsPerBlock(1,1)+patchSize(1,1)-2;
        y=j:j+threadsPerBlock(1,2)+patchSize(1,2)-2;
	
        f = (feval(t,Ad,Ad(x,y),Zd,H,f, m, n,patchSize(1,1),filtSigma));
	wait(gpuDevice);

     end
  end
  f=gather(f);

%remove padding
f=f((1+(patchSize(1,1)-1)/2):(m-(patchSize(1,1)-1)/2),(1+(patchSize(1,1)-1)/2):(m-(patchSize(1,1)-1)/2));
  %% SANITY CHECK
  
  %fprintf('Error: %e\n', norm( B - (A+1), 'fro' ) );
  
  %% (END)

  fprintf('...end %s...\n',mfilename);
toc
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
