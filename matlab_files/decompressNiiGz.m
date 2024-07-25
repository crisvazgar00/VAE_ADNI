%{

Script for decompressing .nii.gz into .nii 
for a root folder with subfolders.
    Requirements: 7-zip file manager (can be changed to 
any other decompressing app)

%}

function decompressNiiGz(fileList)

    for k = 1:length(fileList)

        gzFilePath = fileList{k};
        [folder, baseFilename, ext] = fileparts(gzFilePath);

        if strcmp(ext, '.gz')
                niiFileName = baseFilename; % Remove '.gz' to get the .nii file name
                niiFilePath = fullfile(folder, niiFileName); %generate path of corresponding.nii

            if exist(niiFilePath, 'file') == 2 %Check if that .nii is already decompressed (if it exists)
                    fprintf('File is already decompressed: %s\n', niiFilePath);
            else
                    % Decompress the .nii.gz file
                    gunzip(gzFilePath);
                    fprintf('Decompressed: %s\n', gzFilePath);
            end
        else
            fprintf('File does not have .gz extension: %s\n', gzFilePath);
        end
    end
end


