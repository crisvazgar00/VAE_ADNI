    

    %{

    Script for finding _pet.nii files inside a main_folder. 

    ARGUMENTS:
    
    -Main folder: -Chosen by hand when running script- folder that contains
    all subfolders and files where data is stored.
    
    -Prefix: prefix of the file we want to find. Usually 'sub', 'wsub',
    'rsub', etc.
    
    -Extension: Always set to '.nii' -NIfTI format-

    -targetFolderName: Name of folder where target files are contained.
    This is optional if we set Extension to be '_pet.nii' (for PET) or
    '_T1w.nii' (for MRI).

    %}
    
    
    
    function fileList = find_Pet(folder, prefix, extension, targetFolderName)
         fileList = {}; %List that will contain the file paths
         %Get current Folder
         [~, currentFolderName] = fileparts(folder); %Returns name of the current folder
     
         %Check if current folder coincides with target folder ('pet')
         if strcmp(currentFolderName, targetFolderName)
             %fullfile() gives name of the files with path folder/*extension
             files = dir(fullfile(folder, [prefix, '*', extension])); %dir returns the list of files with that path
             for k = 1:length(files) %loop over every file in the folder
                     fileList{end+1} = fullfile(folder, files(k).name); %add file to list          
             end
         end
     
         %Look inside sub-folders
         subFolders = dir(fullfile(folder, '*')); %Returns every element inside of currentFolder
         %Filters subfolders to check which elements are directories and
         %don't have the name '.' or '..
         subFolders= subFolders([subFolders.isdir] & ~ismember({subFolders.name},{'.', '..'}));
         for i = 1:length(subFolders) %For each subfolder, call the function to
             %check if there are '.nii' files inside
             subFolderFiles = find_Pet(fullfile(folder, subFolders(i).name), prefix, extension, targetFolderName);
             fileList = [fileList, subFolderFiles]; % Add files retrieved
         end
     end
    


    
    
    
    
    

