% Main script for creating list with PET data


%Main folder path
main_folder = fullfile(getenv('Cristobal'), 'Desktop', 'DATOS_PRUEBA', 'ADNI_BIDS');
topLevelFolder = uigetdir(main_folder);


%Call recursive functions with target subfolder name and extension .nii
targetFolderName = 'pet';
extension = '.nii';
listOfFiles = find_Pet(topLevelFolder, extension, targetFolderName); %Call


%Show the amount of files retrieved and the list
numberOfFiles = length(listOfFiles);
disp(['Total number of files retrieved: ', num2str(numberOfFiles)]);
disp('List of files retrieved:');
disp(listOfFiles);