%% read in files
CT1 = double(niftiread('Imaging/Patient-007/week-089/DeepBraTumIA-segmentation/atlas/skull_strip/ct1_skull_strip.nii.gz'))/255;
T1 = double(niftiread('Imaging/Patient-007/week-089/DeepBraTumIA-segmentation/atlas/skull_strip/t1_skull_strip.nii.gz'))/255;
T2 = double(niftiread('Imaging/Patient-007/week-089/DeepBraTumIA-segmentation/atlas/skull_strip/t2_skull_strip.nii.gz'))/255;
FLAIR = double(niftiread('Imaging/Patient-007/week-089/DeepBraTumIA-segmentation/atlas/skull_strip/flair_skull_strip.nii.gz'))/255;

CT1_156 = double(niftiread('Imaging/Patient-007/week-105/DeepBraTumIA-segmentation/atlas/skull_strip/ct1_skull_strip.nii.gz'))/255;
% T1_156 = double(niftiread('Imaging/Patient-067/week-156/DeepBraTumIA-segmentation/atlas/skull_strip/t1_skull_strip.nii.gz'))/255;

% begin registration
[optimizer,metric] = imregconfig("multimodal");
optimizer.InitialRadius = 0.001;

registeredCT1 = imregister(CT1, CT1_156, "affine", optimizer, metric);
registeredT1 = imregister(T1, CT1_156, "affine", optimizer, metric);
registeredT2 = imregister(T2, CT1_156, "affine", optimizer, metric);
registeredFLAIR = imregister(FLAIR, CT1_156, "affine", optimizer, metric);

subplot 141
imshowpair(registeredCT1(:,:,100), CT1_156(:,:,100))
subplot 142
imshowpair(registeredCT1(:,:,100), registeredT1(:,:,100))
subplot 143
imshowpair(registeredCT1(:,:,100), registeredT2(:,:,100))
subplot 144
imshowpair(registeredCT1(:,:,100), registeredFLAIR(:,:,100))

% tform_124_156 = imregtform(CT1,Rmoving,CT1_156,Rfixed,"affine",optimizer,metric);

% transformed_CT1 = imwarp(CT1,Rmoving,tform_124_156,OutputView=Rfixed);
% transformed_T1 = imwarp(T1,Rmoving,tform_124_156,OutputView=Rfixed);
% transformed_T2 = imwarp(T2,Rmoving,tform_124_156,OutputView=Rfixed);
% transformed_FLAIR = imwarp(FLAIR,Rmoving,tform_124_156,OutputView=Rfixed);
% transformed_seg = imwarp(seg,Rmoving,tform_124_156,OutputView=Rfixed);

niftiwrite(registeredCT1, 'alignment_output/DeepBraTumIA/patient_007/CT1_wk089.nii');
niftiwrite(registeredT1, 'alignment_output/DeepBraTumIA/patient_007/T1_wk089.nii');
niftiwrite(registeredT2, 'alignment_output/DeepBraTumIA/patient_007/T2_wk089.nii');
niftiwrite(registeredFLAIR, 'alignment_output/DeepBraTumIA/patient_007/FLAIR_wk089.nii');

%% show results
% subplot(1,2,1)
% ct1_slice = rot90(flip(transformed_CT1_124(:,:,90)));
% ct1_slice = ct1_slice / max(ct1_slice(:));
% imshow(ct1_slice,[])
% hold on
% seg_slice = rot90(flip(transformed_seg_124(:,:,90)));
% seg_slice = seg_slice / max(seg_slice(:)) * 3;
% seg_slice_color = 0.00 + cat(3, seg_slice==3, seg_slice==2, seg_slice==1);
% seg_im = imshow(seg_slice_color,[]);
% set(seg_im, 'AlphaData', seg_slice*0.1);
% title("Registered Wk 124 CT1 Scan with Segmentation Overlay")
%
% subplot(1,2,2)
% ct1_slice = rot90(flip(CT1_156(:,:,90)));
% ct1_slice = ct1_slice / max(ct1_slice(:));
% imshow(ct1_slice,[])
% hold on
% seg_slice = rot90(flip(seg_156(:,:,90)));
% seg_slice = seg_slice / max(seg_slice(:)) * 3;
% seg_slice_color = 0.00 + cat(3, seg_slice==3, seg_slice==2, seg_slice==1);
% seg_im = imshow(seg_slice_color,[]);
% set(seg_im, 'AlphaData', seg_slice*0.1);
% title("Wk 156 CT1 Scan with Segmentation Overlay")

%% show 3d brains together
% viewerUnregistered = viewer3d(BackgroundColor='black',BackgroundGradient='off');
% volshow(CT1_156, Parent=viewerUnregistered, Colormap=[0 1 0]);
% volshow(CT1_124, Parent=viewerUnregistered, Colormap=[0 0 1]);

%% visualize
% viewerUnregistered = viewer3d(BackgroundColor='black',BackgroundGradient='off');
% volshow(moving_reg1, Parent=viewerUnregistered, Colormap=[0 1 0]);
% volshow(seg_156, Parent=viewerUnregistered, Colormap=[0 0 1]);
