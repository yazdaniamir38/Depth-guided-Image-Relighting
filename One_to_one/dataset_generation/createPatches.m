clc;
close all;
clear variables;

I3Directory='../content/';
I2Directory = '../target/';
I1Directory = '../input/';

trainingDirectory_I3 = '../training_patches_varied_256/results/';
trainingDirectory_I2 = '../training_patches_varied_256/target/';
trainingDirectory_I1 = '../training_patches_varied_256/input/';

patchSize = 256;
sizeBy2 = patchSize/2;
stride = patchSize/2;

files_I2 = dir([I2Directory]);
% files_I1 = dir([I1Directory]);
files_I1 = dir([I1Directory, '*.png']);
files_d=dir([I1Directory, '*.mat']);
files_I2=files_I2(~ismember({files_I2.name},{'.','..'}));
files_I1=files_I1(~ismember({files_I1.name},{'.','..'}));

% files_input = dir([inputDirectory, '*.png']);
% files_I1 = dir([hazeDirectory, '*.png']);

length_files = length(files_I2);
delta = .01;
% input_image = imread([inputDirectory, files_input(1).name]);


string_I2 = 'tar';
string_I1 = 'inp';
string_depth='depth';
string_I3='s';
sign=[-1,1];
k =1;
for i = 1: length_files
    disp(k)
    disp(i)
    I3 = imread([I3Directory, strcat('s_',files_I2(i).name)]);
    I2 = imread([I2Directory, files_I2(i).name]);
    I1 = imread([I1Directory, files_I1(i).name]);
    load([I1Directory,files_d(i).name]);
    d=depth{1,1}.normalized_depth;
    I2  = double(I2);
    I1  = double(I1);
    I3=double(I3);
    d=double(d);
    I2 = I2/255;
    I1 = I1/255;
    I3=I3/255;
    [sizeX, sizeY, sizeZ] = size(I2);


    startX = patchSize;
    endX = sizeX - patchSize;
    startY = patchSize;
    endY = sizeY - patchSize;
    
    cropped_I2 = imresize(I2,[patchSize patchSize]);
    cropped_I1 = imresize(I1,[patchSize patchSize]);
    cropped_I3 = imresize(I3,[patchSize patchSize]);
    cropped_d  = imresize(d,[patchSize patchSize]);
    filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
    filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
    filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
    
    filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
    imwrite(cropped_I3, filenameI3);
    imwrite(cropped_I2, filenameI2);
    imwrite(cropped_I1, filenameI1);
    imwrite(cropped_d, filenamedepth);
    k = k+1;
    
%     cropped_I3_flip = fliplr(cropped_I3);
%     cropped_I2_flip = fliplr(cropped_I2);
%     cropped_I1_flip = fliplr(cropped_I1);
%     cropped_d_flip = fliplr(cropped_d);
%     
%     filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%     filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%     filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%     filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%     imwrite(cropped_I3_flip, filenameI3);
%     imwrite(cropped_I2_flip, filenameI2);
%     imwrite(cropped_I1_flip, filenameI1);
%     imwrite(cropped_d_flip, filenamedepth);
%     k = k+1;
angle=sign(randi([1 2],1,6)).*(10*rand(1,6)+2);
    
    cropped_I3_rot = imrotate(cropped_I3,angle(1),'crop');
    cropped_I2_rot = imrotate(cropped_I2,angle(1),'crop');
    cropped_I1_rot = imrotate(cropped_I1, angle(1),'crop');
    cropped_d_rot = imrotate(cropped_d, angle(1),'crop');
    filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
    filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
    filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
    filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
    imwrite(cropped_I3_rot, filenameI3);
    imwrite(cropped_I2_rot, filenameI2);
    imwrite(cropped_I1_rot, filenameI1);
    imwrite(cropped_d_rot, filenamedepth);
    k = k+1;
    
    cropped_I3_rot = imrotate(cropped_I3,angle(2),'crop');
    cropped_I2_rot = imrotate(cropped_I2,angle(2),'crop');
    cropped_I1_rot = imrotate(cropped_I1, angle(2),'crop');
    cropped_d_rot = imrotate(cropped_d, angle(2),'crop');
    filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
    filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
    filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
    filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
    imwrite(cropped_I3_rot, filenameI3);
    imwrite(cropped_I2_rot, filenameI2);
    imwrite(cropped_I1_rot, filenameI1);
    imwrite(cropped_d_rot, filenamedepth);
    k = k+1;    
    
   cropped_I3_rot = imrotate(cropped_I3,angle(3),'crop');
    cropped_I2_rot = imrotate(cropped_I2,angle(3),'crop');
    cropped_I1_rot = imrotate(cropped_I1, angle(3),'crop');
    cropped_d_rot = imrotate(cropped_d, angle(3),'crop');
    filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
    filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
    filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
    filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
    imwrite(cropped_I3_rot, filenameI3);
    imwrite(cropped_I2_rot, filenameI2);
    imwrite(cropped_I1_rot, filenameI1);
    imwrite(cropped_d_rot, filenamedepth);
    k = k+1;
%   
 cropped_I3_rot = imrotate(cropped_I3,angle(4),'crop');
    cropped_I2_rot = imrotate(cropped_I2,angle(4),'crop');
    cropped_I1_rot = imrotate(cropped_I1, angle(4),'crop');
    cropped_d_rot = imrotate(cropped_d, angle(4),'crop');
    filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
    filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
    filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
    filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
    imwrite(cropped_I3_rot, filenameI3);
    imwrite(cropped_I2_rot, filenameI2);
    imwrite(cropped_I1_rot, filenameI1);
    imwrite(cropped_d_rot, filenamedepth);
    k = k+1;
    
     cropped_I3_rot = imrotate(cropped_I3,angle(5),'crop');
    cropped_I2_rot = imrotate(cropped_I2,angle(5),'crop');
    cropped_I1_rot = imrotate(cropped_I1, angle(5),'crop');
    cropped_d_rot = imrotate(cropped_d, angle(5),'crop');
    filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
    filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
    filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
    filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
    imwrite(cropped_I3_rot, filenameI3);
    imwrite(cropped_I2_rot, filenameI2);
    imwrite(cropped_I1_rot, filenameI1);
    imwrite(cropped_d_rot, filenamedepth);
    k=k+1;
    
     cropped_I3_rot = imrotate(cropped_I3,angle(6),'crop');
    cropped_I2_rot = imrotate(cropped_I2,angle(6),'crop');
    cropped_I1_rot = imrotate(cropped_I1, angle(6),'crop');
    cropped_d_rot = imrotate(cropped_d, angle(6),'crop');
    filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
    filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
    filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
    filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
    imwrite(cropped_I3_rot, filenameI3);
    imwrite(cropped_I2_rot, filenameI2);
    imwrite(cropped_I1_rot, filenameI1);
    imwrite(cropped_d_rot, filenamedepth);
    k = k+1;
 
%     cropped_I3_rot = imrotate(cropped_I3_flip,90);
%     cropped_I2_rot = imrotate(cropped_I2_flip,90);
%     cropped_I1_rot = imrotate(cropped_I1_flip, 90);
%     cropped_d_rot = imrotate(cropped_d_flip, 90);
%     filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%     filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%     filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%     filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%     imwrite(cropped_I3_rot, filenameI3);
%     imwrite(cropped_I2_rot, filenameI2);
%     imwrite(cropped_I1_rot, filenameI1);
%     imwrite(cropped_d_rot, filenamedepth);
%     k = k+1;
%     
%     cropped_I3_rot = imrotate(cropped_I3_flip,180);
%     cropped_I2_rot = imrotate(cropped_I2_flip,180);
%     cropped_I1_rot = imrotate(cropped_I1_flip, 180);
%     cropped_d_rot = imrotate(cropped_d_flip, 180);
%     filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%     filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%     filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%     filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%     imwrite(cropped_I3_rot, filenameI3);
%     imwrite(cropped_I2_rot, filenameI2);
%     imwrite(cropped_I1_rot, filenameI1);
%     imwrite(cropped_d_rot, filenamedepth);
%     k = k+1;
    %     
%     cropped_I3_rot = imrotate(cropped_I3_flip,270);
%     cropped_I2_rot = imrotate(cropped_I2_flip,270);
%     cropped_I1_rot = imrotate(cropped_I1_flip, 270);
%     cropped_d_rot = imrotate(cropped_d_flip, 270);
%     filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%     filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%     filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%     filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%     imwrite(cropped_I3_rot, filenameI3);
%     imwrite(cropped_I2_rot, filenameI2);
%     imwrite(cropped_I1_rot, filenameI1);
%     imwrite(cropped_d_rot, filenamedepth);
%     k = k+1;
%     I1=imresize(I1,[512 512]);
%     I2=imresize(I2,[512 512]);
%     d=imresize(d,[512 512]);
%     angle=sign(randi([1 2],1))*(8*rand(1)+2);
    I3 =  padarray(I3,[sizeBy2,sizeBy2]);
    I2 = padarray(I2, [sizeBy2,sizeBy2]);
    I1 = padarray(I1, [sizeBy2,sizeBy2]);
    d=padarray(d,[sizeBy2,sizeBy2]);
% %     I3r=padarray(imrotate(I3,angle,'crop'),[sizeBy2,sizeBy2]);
%     I2r=padarray(imrotate(I2,angle,'crop'),[sizeBy2,sizeBy2]);
%     I1r=padarray(imrotate(I1,angle,'crop'),[sizeBy2,sizeBy2]);
%     dr=padarray(imrotate(d,angle,'crop'),[sizeBy2,sizeBy2]);
    for x = startX:stride:endX
        for y = startY:stride:endY
            I3_patch = I3(x-sizeBy2+1:x+sizeBy2, y-sizeBy2+1:y+sizeBy2,:);
            I2_patch = I2(x-sizeBy2+1:x+sizeBy2, y-sizeBy2+1:y+sizeBy2,:);
            I1_patch = I1(x-sizeBy2+1:x+sizeBy2, y-sizeBy2 + 1:y+sizeBy2, :);
            l1_nz=length(find(I1_patch>.1));
            l2_nz=length(find(I2_patch>.1));
            if l1_nz/256^2<.1 && l2_nz/256^2<.1
                continue
            end   
            d_patch=d(x-sizeBy2+1:x+sizeBy2, y-sizeBy2 + 1:y+sizeBy2);
            filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
            filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
            filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
            filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
            imwrite(I3_patch, filenameI3);
            imwrite(I2_patch, filenameI2);
            imwrite(I1_patch, filenameI1);
            imwrite(d_patch, filenamedepth);
            k = k+1;
            
%             cropped_I3_flip = fliplr(I3_patch);
%             cropped_I2_flip = fliplr(I2_patch);
%             cropped_I1_flip = fliplr(I1_patch);
%             cropped_d_flip=fliplr(d_patch);
%             filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I3_flip, filenameI3);
%             imwrite(cropped_I2_flip, filenameI2);
%             imwrite(cropped_I1_flip, filenameI1);
%             imwrite(cropped_d_flip, filenamedepth);
%             k = k+1;
%             ind_1=randi(2,1);
%             ind_2=randperm(3,3);
%             switch ind_1
%                 case 1
%             angle=sign(randi([1 2],1,3)).*(8*rand(1,3)+2);
%             cropped_I3_rot = imrotate(I3_patch,angle(1));
%             cropped_I2_rot = imrotate(I2_patch,angle(1));
%             cropped_I1_rot = imrotate(I1_patch, angle(1));
%             cropped_d_rot = imrotate(d_patch,angle(1));
%             filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I3_rot, filenameI3);
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;
% %             
%             cropped_I3_rot = imrotate(I3_patch,angle(2));
%             cropped_I2_rot = imrotate(I2_patch,angle(2));
%             cropped_I1_rot = imrotate(I1_patch, angle(2));
%             cropped_d_rot = imrotate(d_patch, angle(2));
%             filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I3_rot, filenameI3);
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;            
%             
%             cropped_I3_rot = imrotate(I3_patch,angle(3));
%             cropped_I2_rot = imrotate(I2_patch,angle(3));
%             cropped_I1_rot = imrotate(I1_patch, angle(3));
%             cropped_d_rot = imrotate(d_patch, angle(3));
%             filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I3_rot, filenameI3);
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;
%              cropped_I2_rot = imrotate(I2_patch,90*ind_2(3));
%             cropped_I1_rot = imrotate(I1_patch, 90*ind_2(3));
%             cropped_d_rot = imrotate(d_patch, 90*ind_2(3));
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;
%             
%             cropped_I3_rot = imrotate(cropped_I3_flip,90);
%             cropped_I2_rot = imrotate(cropped_I2_flip,90);
%             cropped_I1_rot = imrotate(cropped_I1_flip, 90);
%             cropped_d_rot = imrotate(cropped_d_flip, 90);
%             filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I3_rot, filenameI3);
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;
%                 case 2
%             cropped_I3_rot = imrotate(cropped_I3_flip,180);
%             cropped_I2_rot = imrotate(cropped_I2_flip,180);
%             cropped_I1_rot = imrotate(cropped_I1_flip, 180);
%             cropped_d_rot = imrotate(cropped_d_flip, 180);
%             filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I3_rot, filenameI3);
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;
%             
%             cropped_I3_rot = imrotate(cropped_I3_flip,270);
%             cropped_I2_rot = imrotate(cropped_I2_flip,270);
%             cropped_I1_rot = imrotate(cropped_I1_flip, 270);
%             cropped_d_rot = imrotate(cropped_d_flip, 270);
%             filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I3_rot, filenameI3);
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);%
%             k = k+1;
%             cropped_I2_rot = imrotate(cropped_I2_flip,90*ind_2(3));
%             cropped_I1_rot = imrotate(cropped_I1_flip, 90*ind_2(3));
%             cropped_d_rot = imrotate(cropped_d_flip, 90*ind_2(3));
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;
%             
%             cropped_I2_rot = imrotate(I2_patch,90*ind_2(1));
%             cropped_I1_rot = imrotate(I1_patch, 90*ind_2(1));
%             cropped_d_rot = imrotate(d_patch, 90*ind_2(1));
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             imwrite(cropped_d_rot,filenamedepth);
%             k = k+1;
%             
%             end
%             cropped_I2_rot = imrotate(I2_patch,270);
%             cropped_I1_rot = imrotate(I1_patch, 270);
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             k = k+1;
%             
%             
%             cropped_I2_rot = imrotate(cropped_I2_flip,90);
%             cropped_I1_rot = imrotate(cropped_I1_flip, 90);
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             k = k+1;
%             
%             cropped_I2_rot = imrotate(cropped_I2_flip,180);
%             cropped_I1_rot = imrotate(cropped_I1_flip, 180);
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             k = k+1;
%             
%             cropped_I2_rot = imrotate(cropped_I2_flip,270);
%             cropped_I1_rot = imrotate(cropped_I1_flip, 270);
%             filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
%             filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
%             imwrite(cropped_I2_rot, filenameI2);
%             imwrite(cropped_I1_rot, filenameI1);
%             k = k+1;
        end
    end
    for a=1:15
    angle=sign(randi([1 2],1))*(10*rand(1)+2);
    I3r=padarray(imrotate(I3,angle,'crop'),[sizeBy2,sizeBy2]);
    I2r=padarray(imrotate(I2,angle,'crop'),[sizeBy2,sizeBy2]);
    I1r=padarray(imrotate(I1,angle,'crop'),[sizeBy2,sizeBy2]);
    dr=padarray(imrotate(d,angle,'crop'),[sizeBy2,sizeBy2]);
    
for x = startX+stride:stride:endX-stride
        for y = startY+stride:stride:endY-stride
            I3_patch = I3r(x-sizeBy2+1:x+sizeBy2, y-sizeBy2+1:y+sizeBy2,:);
            I2_patch = I2r(x-sizeBy2+1:x+sizeBy2, y-sizeBy2+1:y+sizeBy2,:);
            I1_patch = I1r(x-sizeBy2+1:x+sizeBy2, y-sizeBy2 + 1:y+sizeBy2, :);
            l1_nz=length(find(I1_patch>.1));
            l2_nz=length(find(I2_patch>.1));
            if l1_nz/256^2<.1 && l2_nz/256^2<.1
                continue
            end   
            d_patch=dr(x-sizeBy2+1:x+sizeBy2, y-sizeBy2 + 1:y+sizeBy2);
            filenameI3 = [trainingDirectory_I3,string_I3,'_',num2str(k),'.png'];
            filenameI2 = [trainingDirectory_I2,string_I2,'_',num2str(k),'.png'];
            filenameI1 = [trainingDirectory_I1,string_I1,'_',num2str(k),'.png'];
            filenamedepth = [trainingDirectory_I1,string_depth,'_',num2str(k),'.png'];
            imwrite(I3_patch, filenameI3);
            imwrite(I2_patch, filenameI2);
            imwrite(I1_patch, filenameI1);
            imwrite(d_patch, filenamedepth);
            k = k+1;    
        end
end
    end
end

