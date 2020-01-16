% test proj
clear;
close all;


%%adjust to your actual location before testing
caffe_folder = '../../';
addpath '../../../../matlab';

caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(2);


loss_test=zeros(100,3);

model = './MC_RDN_net_deploy.prototxt';
weights = './Model/paper_model.caffemodel';
net = caffe.Net(model, weights, 'test');

localmap=zeros(720,1280,2);

for i=1:1280
    for j=1:720
        localmap(j,i,1)=i/1280;
        localmap(j,i,2)=j/720;
    end
end

PSNR_DL_perpic=zeros(13*100,1);
SSIM_DL_perpic=zeros(13*100,1);

for j=0:1:12
    j+1

    PSNR_ALL=0.0;
    SSIM_ALL=0.0;
    for i=1+100*j:100+100*j

        im_ori = imread(strcat('../dataset/test/inf/',num2str(i),'.png'));
        im_ori = im2double(im_ori);

        
        im_proj_ori = imread(strcat('../dataset/test/def/',num2str(i),'.png'));
        im_proj_ori = im2double(im_proj_ori);
        
        img_dep= imread(strcat('../dataset/depth/',num2str(uint16(ceil(i/100))),'.png'));
        img_dep=(double(img_dep))/1300;
        
        
        im_ori(:,:,6)=img_dep;
        im_ori(:,:,4:5)=localmap;
        im_ori_temp=im_ori;

        [H, W, C] = size(im_ori);
        im_ori = single(im_ori);
        net.blobs('data').reshape([H W C 1]);

        tic;
        output = net.forward({im_ori});
	toc;

        im_proj = output{1};
    
    	%According to the extreme coordinate of corner points(40:680,40:1240) for warpping,
	%to avoid the edge pixels of the images which are excessively intensity enlighted,	
	%image are cropped before evalutaion   
        PSNR=psnr(double(im_proj(40:680,40:1240,:)),im_proj_ori(40:680,40:1240,:));
        PSNR_DL_perpic(i)=PSNR;
        PSNR_ALL=PSNR_ALL+PSNR;
        
        SSIM=psnr(double(im_proj(40:680,40:1240,:)),im_proj_ori(40:680,40:1240,:));
        SSIM_DL_perpic(i)=SSIM;
        SSIM_ALL=SSIM_ALL+SSIM;

    end
    
    PSNR_ALL/(100*1)
    SSIM_ALL/100
end

