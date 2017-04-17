close all;clear all;clc

%%
load digitStruct.mat
len = length(digitStruct);
y = 10*ones(len, 6);
for i = 1:len
    im = imread(digitStruct(i).name);
    X(:,:,:,i) = imresize(im, [128, 128]);
    y(i, 1) = length(digitStruct(i).bbox);
    if length(digitStruct(i).bbox)<6
        for j = 1:length(digitStruct(i).bbox)
            y(i, j+1) = mod(digitStruct(i).bbox(j).label, 10);
        end
    else
        for j = 1:5
            y(i, j+1) = mod(digitStruct(i).bbox(j).label, 10);
        end
    end
end

save train_128x128.mat X y