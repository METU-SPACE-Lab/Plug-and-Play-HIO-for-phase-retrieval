clc
clear
restoredefaultpath;
addpath(genpath('./'));
% reset(gpuDevice(1))
run ./Packages/matconvnet-1.0-beta23/matlab/vl_setupnn

test_images_dir = dir('./test_images/*.png');
out_images_dir = './output_images/';
mkdir(out_images_dir);

totalIter = 200;
iterhio = 5;
modelSigma1 = 40;
modelSigma2 = 5;
modelSigmaS = logspace(log10(modelSigma1),log10(modelSigma2),totalIter);

load('./Packages/modelgray.mat');
ns          = min(25,max(ceil(modelSigmaS/2),1));
ns          = [ns(1)-1,ns];
cmode='cpu';
useGPU       = 0;
alpha=3; %noise level

for k = 1:length(test_images_dir)


img = single(imread(fullfile(test_images_dir(k).folder,test_images_dir(k).name)));
if min(min(img))<0
x_0 =double(norm255(img));
else
x_0 =double(img);  
end
x_0255 = norm255(x_0);
%%
   
[height, width]=size(x_0);
SamplingRate=4;
n=length(x_0(:));
m=round(n*SamplingRate);%May be overwritten when using Fourier measurements
I=speye(m);
mask=padarray(ones(height,width),[1/2*(sqrt(m)-sqrt(n)),1/2*(sqrt(m)-sqrt(n))]);
OversampM=I(:,logical(mask(:)));    

imsize=height; 
cor_support = 2*imsize;
re_cor_support = cor_support;

M=@(z) reshape(fft2(reshape(z,[size(mask,1),size(mask,2)]),re_cor_support,re_cor_support),[m,1])*(1/sqrt(m))*sqrt(n/m);%x must be
Mt=@(z) reshape((ifft2(reshape(z(:),[size(mask,1),size(mask,2)]),'symmetric')),[m,1])*sqrt(m)*sqrt(m/n);

M2=@(x) reshape((fft2(reshape(OversampM*x(:),[sqrt(m/n)*height,sqrt(m/n)*width]))),[m,1])*(1/sqrt(m))*sqrt(n/m);
Mt2=@(z) OversampM'*reshape((ifft2(reshape(z(:),[sqrt(m/n)*height,sqrt(m/n)*width]))),[m,1])*sqrt(m)*sqrt(n/m);


SNR=@(a,b) 10*log10(norm(a)/norm(a-b));

%% HIO Parameters
support=mask;
beta=.9;
HIO_init_iters=5e1;

HIO_iters_hio=1e3;

Pm=@(x,y) Mt(y.*exp(1i*angle(M(x))));
%%

z=M2(x_0(:));
intensity_noise=alpha*abs(z).*randn(m,1);
z2=abs(z).^2;
y2=abs(z).^2+intensity_noise;
y2=y2.*(y2>0);
y=sqrt(y2);

err=y(:)-abs(z);
sigma_w=std(err);

SNR_diff_sq = SNR(double(norm255(z2)),double(norm255(y2)));

%% HIO
% 
if strcmp(cmode,'gpu')
    y_gpu = gpuArray(y(:));
    x_init = gpuArray(rand(m,1));
    x_init(1)=1;
else
    y_gpu = y(:);  
    x_init = rand(m,1);
    x_init(1)=1;
end

resid_best=inf;
t0=tic;
x_init_best=nan(sqrt(n),sqrt(n));
for j=1:50
    x_init_i=HIO( y_gpu, M, Mt, support(:), beta,HIO_init_iters,x_init,cmode );
    x_init_i = gather(x_init_i);
    resid_i=norm(y-abs(M(x_init_i)));
    if resid_i<resid_best
        resid_best=resid_i;
        x_init_best=x_init_i;
    end
end

if strcmp(cmode,'gpu')
    x_init_best_gpu = gpuArray(x_init_best);
else
    x_init_best_gpu = x_init_best; 
end
x_hat_HIO=HIO( y_gpu, M, Mt, support(:), beta,HIO_iters_hio,x_init_best_gpu, cmode );
t_HIO=toc(t0);
 x_hat_HIO = gather(x_hat_HIO);
 x_hat_HIO = OversampM'*real(x_hat_HIO(:));
 alpha3 = (x_hat_HIO'*x_0(:))/(x_hat_HIO'*x_hat_HIO);
 tav = rot90(reshape(x_hat_HIO,[height,width]),2);
 alpha2 = (tav(:)'*x_0(:))/(tav(:)'*tav(:));
 if alpha2>alpha3
     x_hat_HIO = tav(:);
 end
x_hat_HIO = reshape(x_hat_HIO,[height,width]);


%% 

%% Developed method

if strcmp(cmode,'gpu')
    x_init_i = gpuArray(OversampM*real(x_hat_HIO(:)/255));
else
    x_init_i = OversampM*real(x_hat_HIO(:)/255);

end
%figure,imshow(x_hat_HIO,[])
y_gpu_old=y_gpu/255;
t1=tic;


SSIM_HIO = ssim(single(x_hat_HIO./255),single(x_0./255));
PSNR_HIO = PSNR(single(x_hat_HIO),single(x_0));


for itern=1:totalIter
    x_init_i_sqr = OversampM'*real(x_init_i(:));
    x_init_i_sqr = reshape((x_init_i_sqr),[imsize,imsize]);
    
    y1=abs(M(x_init_i(:)));
    %kaf1 = max(modelSigmaS(itern)/modelSigma1,0.15/sigma_w);
    kaf1 = modelSigmaS(itern)/modelSigma1;
    y = y_gpu_old*kaf1+y1(:)*(1-kaf1); 
   
  
    for kaf=1:iterhio
        Pmx=Pm(x_init_i,y);
        inds=logical(isreal(Pmx).*support(:).*(real(Pmx)>0));
        x_init_i(inds)=Pmx(inds);
        x_init_i(~inds)=x_init_i(~inds)-beta*Pmx(~inds);
    end
    x_hat_HIO_k = OversampM'*real(x_init_i(:));
    x_hat_HIO_r = reshape((x_hat_HIO_k),[imsize,imsize]);
    if ns(itern+1)~=ns(itern)
        [net1] = loadmodel(modelSigmaS(itern),CNNdenoiser);
        net1 = vl_simplenn_tidy(net1);
        if strcmp(cmode,'gpu')
            net1 = vl_simplenn_move(net1, 'gpu');
        end
    end
    res = vl_simplenn(net1, (single(x_hat_HIO_r)),[],[],'conserveMemory',true,'mode','test');
    residual = res(end).x;
    x_init_i_dum= (x_hat_HIO_r) - residual;
    
    diff = x_init_i_sqr-x_init_i_dum;
    
    x_init_i = x_init_i_dum;
    xf = x_init_i;
    

            
    all_diff(itern) = norm(diff)/norm(x_init_i);
    
    x_init_i = OversampM*real(double((x_init_i(:))));
end

HIO_out = x_hat_HIO;
PPHIO_out=gather(xf);

PSNR_PPHIO = PSNR((single(255*PPHIO_out)),(single(x_0)));
SSIM_PPHIO = ssim(single(PPHIO_out),single(x_0./255));

disp(['PSNR-HIO: ',num2str(PSNR_HIO)]);
disp(['SSIM-HIO: ',num2str(SSIM_HIO)]);

disp(['PSNR-PPHIO: ',num2str(PSNR_PPHIO)]);
disp(['SSIM-PPHIO: ',num2str(SSIM_PPHIO)]);

imwrite(uint8(norm255(x_0)),fullfile(out_images_dir,[num2str(k),'_original.png']))
imwrite(uint8(norm255(HIO_out)),fullfile(out_images_dir,[num2str(k),'_HIO_out.png']))
imwrite(uint8(norm255(PPHIO_out)),fullfile(out_images_dir,[num2str(k),'_PPHIO_out.png']))
% 
% figure,
% imshow(norm255(x_0),[]);
% title('Original')
% figure,
% imshow(norm255(HIO_out),[]);
% title('HIO method')
% figure,
% imshow(norm255(PPHIO_out),[]);
% title('Developed method')
% % % % 

end

