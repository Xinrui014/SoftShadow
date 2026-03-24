%% compute RMSE(MAE)|计算RMSE(MAE) 
clear all;close all;clc
% 1`modify the following directories 2`run|修改路径,再运行

% GT mask directory|掩膜路径
maskdir = '~/Desktop/projects/sam_shadow_removal/SRD/srd_mask_DHAN/SRD_testmask';
MD = dir([maskdir '/*.jpg']);

% result directory|结果路径
shadowdir = '~/Desktop/projects/sam_shadow_removal/ShadowDiffusion/results/official_eval_samples/results';  %test_A

SD_f = dir([shadowdir '/*']);
filteredFileList = {};
% Loop through each filename in the list
for k = 1:length(SD_f)
    % Get the current filename
    filename = SD_f(k).name;
    
    % Check if the filename ends with '_sr_process.png'
    if endsWith(filename, '_sr_process.mat')
        % If it does, add it to the filtered list
        filteredFileList{end+1} = filename; %#ok<SAGROW>
    end
end

SD=filteredFileList;

% ground truth directory|GT路径
% freedir = '~/Desktop/projects/sam_shadow_removal/SRD/test/shadow_free'; %AISTD
% FD = dir([freedir '/*.jpg']);



total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
allmae=zeros(1,size(MD,1)); 
smae=zeros(1,size(MD,1)); 
nmae=zeros(1,size(MD,1)); 
ppsnr=zeros(1,size(MD,1));
ppsnrs=zeros(1,size(MD,1));
ppsnrn=zeros(1,size(MD,1));
sssim=zeros(1,size(MD,1));
sssims=zeros(1,size(MD,1));
sssimn=zeros(1,size(MD,1));
cform = makecform('srgb2lab');

for i=1:size(MD)
    %disp(SD(i));
    %disp(FD(i));
    %disp(MD(i));
    
    [~, name, ~] = fileparts(SD{i});
    fname = fullfile(shadowdir, sprintf('%s_hr.mat', name(1:end-11)));
    sname = fullfile(shadowdir, SD{i});
    parts = strsplit(name,"_");
    if strcmp(parts{1}, '')
    % Keep the leading underscore and reconstruct the string without the last three parts
        new_name = ['_', strjoin(parts(2:end-3), '_')];
    else
    % Reconstruct the string without the last three parts
        new_name = strjoin(parts(1:end-3), '_');
    end
    name = new_name;
%     name = 'IMG_1_5453';
    mname = fullfile(maskdir, sprintf('%s.jpg', name));

    

%     s=imread(sname);
    s=load(sname);
    f=load(fname);
    
    try
        % Attempt to read the image
        
%         f=imread(fname);
        m=imread(mname);
        % If successful, the following code will execute
        % Process your image here
        
    catch ME
        % If an error occurred, check the identifier
        if strcmp(ME.identifier, 'MATLAB:imagesci:imread:fileDoesNotExist')
            fprintf('File not found: %s\n', fname);
            continue; % Skip to the next iteration of the loop
        else
            rethrow(ME); % If it is a different kind of error, rethrow it
        end
    end
    
%     f=imread(fname);
%     m=imread(mname);
    
    f = double(f.img)/255;
    s = double(s.img)/255;

    
    s=imresize(s,[256 256]);
    f=imresize(f,[256 256]);
    m=imresize(m,[256 256]);


    nmask=~m;       %mask of non-shadow region|非阴影区域的mask
    smask=~nmask;   %mask of shadow regions|阴影区域的mask
    
    ppsnr(i)=psnr(s,f);
    ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    sssim(i)=ssim(s,f);
    sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));

    f = applycform(f,cform);    
    s = applycform(s,cform);
    
    %% MAE, per image
    dist=abs((f - s));
    sdist=dist.*repmat(smask,[1 1 3]);
    sumsdist=sum(sdist(:));
    ndist=dist.*repmat(nmask,[1 1 3]);
    sumndist=sum(ndist(:));
    
    sumsmask=sum(smask(:));
    sumnmask=sum(nmask(:));
    
    %% MAE, per pixel
    allmae(i)=sum(dist(:))/size(f,1)/size(f,2);
    smae(i)=sumsdist/sumsmask;
    nmae(i)=sumndist/sumnmask;
    
    total_dists = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    
    total_distn = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;  

%     disp(i);
end
% a = ppsnr;
% fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr ~= 0),mean(ppsnrn),mean(ppsnrs));
% fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
% fprintf('PI-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(allmae),mean(nmae),mean(smae));
% fprintf('PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(allmae),total_distn/total_pixeln,total_dists/total_pixels);

% Calculate means excluding zero values
a = ppsnr(ppsnr ~= 0);
ppsnr_mean = mean(ppsnr(ppsnr ~= 0));
ppsnrn_mean = mean(ppsnrn(ppsnrn ~= 0));
ppsnrs_mean = mean(ppsnrs(ppsnrs ~= 0));

sssim_mean = mean(sssim(sssim ~= 0));
sssimn_mean = mean(sssimn(sssimn ~= 0));
sssims_mean = mean(sssims(sssims ~= 0));

allmae_mean = mean(allmae(allmae ~= 0));
nmae_mean = mean(nmae(nmae ~= 0));
smae_mean = mean(smae(smae ~= 0));

% Print results
fprintf('PSNR (all, non-shadow, shadow): %f\t%f\t%f\n', ppsnr_mean, ppsnrn_mean, ppsnrs_mean);
fprintf('SSIM (all, non-shadow, shadow): %f\t%f\t%f\n', sssim_mean, sssimn_mean, sssims_mean);
fprintf('PI-Lab (all, non-shadow, shadow): %f\t%f\t%f\n', allmae_mean, nmae_mean, smae_mean);
fprintf('PP-Lab (all, non-shadow, shadow): %f\t%f\t%f\n', allmae_mean, total_distn/total_pixeln, total_dists/total_pixels);