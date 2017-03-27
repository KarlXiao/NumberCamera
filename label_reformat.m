close all;clear all;clc
load('test_32x32.mat')
for i=1:length(y)
   if y(i)==10
       y(i)=0;
   end
end
save new_test_32x32.mat X y

load('train_32x32.mat')
for i=1:length(y)
   if y(i)==10
       y(i)=0;
   end
end
save new_train_32x32.mat X y