


%{

 *This script allows for Normalisation (Est + Writ) to MNI space with SPM12 
 & Segmentation (WM, GM, CSF, SKULL, REST).

 *We use the functon find_Pet to obtain every _pet.nii file  and iterate .

 *To just apply Normalization set 'mode' to 'Norm'. For Segmentation 'Seg'.
 To apply both preprocessing techniques set 'mode' to 'both'.

 *Do NOT change extension '.nii' since it is used in every file for SPM.

 *Prefix 'sub'    ----> raw data.
  Prefix 'wsub'   ----> Normalised (Est+Writ) data
  Prefix 'rsub'   ----> Corregistered data (to MNI)

 *If raw data is not decompressed (.nii.gz is compressed), set compressed
 to true. Decompression is performed using 'decompressNiiGz.m file

%}


clear all;
clc;

compressed = false;

% Norm, Seg, Correg, All
mode = 'Correg';

%Choose main folder (where patient data is stored)
main_folder = fullfile(getenv('Cristobal'), 'Desktop', 'ADNI_BIDS');
topLevelFolder = uigetdir(main_folder);

%Call recursive functions with target subfolder name 'pet' and extension
%'.nii'
targetFolderName = 'pet';

spm('defaults', 'FMRI'); % Ensure SPM is initialized correctly
spm_jobman('initcfg');
matlabbatch = []; %This will contain the batch of images to preprocess


if(compressed == true)
    prefix = 'sub';
    extension = '.nii.gz';
    compressedFiles = find_Pet(topLevelFolder, prefix, extension, targetFolderName);
    decompressNiiGz(compressedFiles);
    fprintf('== Decompression completed succesfully ==')
end

extension = '.nii'; %Set extension to .nii once uncompressed

%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOOP FOR NORMALISATION %
%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(mode, 'Norm') || strcmp(mode, 'All')
    %Find files with extension '.nii' & prefix 'sub'
    prefix = 'sub';
    
    listOfFiles = find_Pet(topLevelFolder, prefix, extension, targetFolderName); 

     % Check if listOfFiles is empty
    if isempty(listOfFiles)
        error(sprintf('No files found with the specified prefix "%s" and extension "%s".', prefix, extension));
    else
    
        %For debugging purposed
        disp('List of Files');
        disp(listOfFiles);
        
        %If not empty then load files and run batch

        disp('Normalisation initiated');
        
        for k = 1:length(listOfFiles)
            matlabbatch{k}.spm.spatial.normalise.estwrite.subj.vol = {listOfFiles{k}};
            matlabbatch{k}.spm.spatial.normalise.estwrite.subj.resample = {listOfFiles{k}};
            matlabbatch{k}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
            matlabbatch{k}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
            matlabbatch{k}.spm.spatial.normalise.estwrite.eoptions.tpm = {'C:\Program Files\spm12\tpm\TPM.nii'};
            matlabbatch{k}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
            matlabbatch{k}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
            matlabbatch{k}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
            matlabbatch{k}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
            matlabbatch{k}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
                                                             78 76 85];
            matlabbatch{k}.spm.spatial.normalise.estwrite.woptions.vox = [1.5 1.5 1.5]; %MNI Space voxel size is 1.5mm
            matlabbatch{k}.spm.spatial.normalise.estwrite.woptions.interp = 4;
            matlabbatch{k}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';
        end 

        spm_jobman('run', matlabbatch) %Run the preprocessing batch
        disp('Normalisation terminated');
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%
% LOOP FOR SEGMENTATION %
%%%%%%%%%%%%%%%%%%%%%%%%%



if strcmp(mode, 'Seg') || strcmp(mode, 'All')
    spm_jobman('initcfg');
    matlabbatch = []; %This will contain the batch of images to preprocess
    listOfFiles = {};

    %Find files with extension '.nii' & prefix 'wsub'
    prefix = 'wsub';
    listOfFiles = find_Pet(topLevelFolder, prefix, extension, targetFolderName); 

    % Check if listOfFiles is empty
    if isempty(listOfFiles)
        error(sprintf('No files found with the specified prefix "%s" and extension "%s".', prefix, extension));
    else
        
        %For debugging purposed
        disp('List of Files');
        disp(listOfFiles);

        %If not empty then load and run the batch
        disp('Segmentation initiated');

        for k = 1:length(listOfFiles)
            matlabbatch{k}.spm.spatial.preproc.channel.vols = {listOfFiles{k}};
            matlabbatch{k}.spm.spatial.preproc.channel.biasreg = 0.001;
            matlabbatch{k}.spm.spatial.preproc.channel.biasfwhm = 60;
            matlabbatch{k}.spm.spatial.preproc.channel.write = [0 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(1).tpm = {'C:\Program Files\spm12\tpm\TPM.nii,1'};
            matlabbatch{k}.spm.spatial.preproc.tissue(1).ngaus = 1;
            matlabbatch{k}.spm.spatial.preproc.tissue(1).native = [1 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(1).warped = [0 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(2).tpm = {'C:\Program Files\spm12\tpm\TPM.nii,2'};
            matlabbatch{k}.spm.spatial.preproc.tissue(2).ngaus = 1;
            matlabbatch{k}.spm.spatial.preproc.tissue(2).native = [1 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(2).warped = [0 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(3).tpm = {'C:\Program Files\spm12\tpm\TPM.nii,3'};
            matlabbatch{k}.spm.spatial.preproc.tissue(3).ngaus = 2;
            matlabbatch{k}.spm.spatial.preproc.tissue(3).native = [1 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(3).warped = [0 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(4).tpm = {'C:\Program Files\spm12\tpm\TPM.nii,4'};
            matlabbatch{k}.spm.spatial.preproc.tissue(4).ngaus = 3;
            matlabbatch{k}.spm.spatial.preproc.tissue(4).native = [1 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(4).warped = [0 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(5).tpm = {'C:\Program Files\spm12\tpm\TPM.nii,5'};
            matlabbatch{k}.spm.spatial.preproc.tissue(5).ngaus = 4;
            matlabbatch{k}.spm.spatial.preproc.tissue(5).native = [1 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(5).warped = [0 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(6).tpm = {'C:\Program Files\spm12\tpm\TPM.nii,6'};
            matlabbatch{k}.spm.spatial.preproc.tissue(6).ngaus = 2;
            matlabbatch{k}.spm.spatial.preproc.tissue(6).native = [0 0];
            matlabbatch{k}.spm.spatial.preproc.tissue(6).warped = [0 0];
            matlabbatch{k}.spm.spatial.preproc.warp.mrf = 1;
            matlabbatch{k}.spm.spatial.preproc.warp.cleanup = 1;
            matlabbatch{k}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
            matlabbatch{k}.spm.spatial.preproc.warp.affreg = 'mni';
            matlabbatch{k}.spm.spatial.preproc.warp.fwhm = 0;
            matlabbatch{k}.spm.spatial.preproc.warp.samp = 3;
            matlabbatch{k}.spm.spatial.preproc.warp.write = [0 0];
            matlabbatch{k}.spm.spatial.preproc.warp.vox = NaN;
            matlabbatch{k}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
        end

        spm_jobman('run', matlabbatch) %Run the preprocessing batch
        disp('Segmentation terminated');
    end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%
% CORREGISTRATION TO MNI %
%%%%%%%%%%%%%%%%%%%%%%%%%%


if strcmp(mode, 'Correg') || strcmp(mode, 'All')
    spm_jobman('initcfg');
    matlabbatch = []; %This will contain the batch of images to preprocess
    listOfFiles = {};

    %Find files with extension '.nii' & prefix 'sub'
    prefix = 'sub';
    listOfFiles = find_Pet(topLevelFolder, prefix, extension, targetFolderName); 

    % Check if listOfFiles is empty
    if isempty(listOfFiles)
        error(sprintf('No files found with the specified prefix "%s" and extension "%s".', prefix, extension));
    else

        spm_progress_bar('Init', length(listOfFiles), 'Co-registering Images', 'Files processed');

        %If not empty then load and run the batch
        disp('Corregistration initiated');

        for k = 1:length(listOfFiles)
            matlabbatch{k}.spm.spatial.coreg.estwrite.ref = {'C:\Program Files\spm12\canonical\single_subj_T1.nii,1'};
            matlabbatch{k}.spm.spatial.coreg.estwrite.source = {listOfFiles{k}};
            matlabbatch{k}.spm.spatial.coreg.estwrite.other = {''};
            matlabbatch{k}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
            matlabbatch{k}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
            matlabbatch{k}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
            matlabbatch{k}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
            matlabbatch{k}.spm.spatial.coreg.estwrite.roptions.interp = 4;
            matlabbatch{k}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
            matlabbatch{k}.spm.spatial.coreg.estwrite.roptions.mask = 0;
            matlabbatch{k}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

            % Update progress bar
            spm_progress_bar('Set', k);
        end

        spm_jobman('run', matlabbatch) %Run the preprocessing batch
        disp('Segmentation terminated');
        
        % Close progress bar
        spm_progress_bar('Clear');
    end

end



