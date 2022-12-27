clc;
clear all %clears memory
%Name: Adigun Peter Oluwasayo
%Class: CS 4730-5730, Artificial Inteligence
%Date: 4-22-2021
%Title:FFNN,4x4x1 Model of Iris Data Set
%Notes: Version 1 of 1

%Load our data, X_iris
X_iris = csvread('iris.csv');
X_iris = X_iris(:,1:size(X_iris,2)-1);
%Feautures
% x_0 = sepal length, x_1 = sepal width,
% x_2 = petal length, x_3 = petal width.

for X = 1:size(X_iris,2)
  X1_hat(:,X) = Normalization_1(X_iris(:,X));
  X2_hat(:,X) = Norm_2(X_iris(:,X));
  X3_hat(:,X) = Normalization_3(X_iris(:,X));
end
%Generate the Label
y_l = zeros(size(X2_hat,1),3);
y_l(1:50,1) = ones (50,1);
y_l(51:100,2) = ones (50,1);
y_l(101:150,3) = ones (50,1);


D_iris1 = [X1_hat y_l];
D_iris2 = [X2_hat y_l];
D_iris3 = [X3_hat y_l];


%for loop with k=folds
%Randomize D_iris and Partition D_iris_train (75%)
%D_iris_test (25%) => D_iris_train => 90 instances
%D_iris_test => 10 instances
new_index = randperm(size(D_iris1,1),size(D_iris1,1));
for i=1:size(D_iris1,1)
  index = new_index(i);
  D_iris1(index,:) = D_iris1(i,:);
endfor
  
D_iris1_train = D_iris1(1:uint64(size(D_iris1,1).*0.75),:);
D_iris1_test = D_iris1(uint64(size(D_iris1,1).*0.75)+1:size(D_iris1,1),:);

Print2file_OHE(D_iris1_train,'D_iris3_train0.csv')%1- 100
Print2file_OHE(D_iris1_test,'D_iris3_test0.csv')%1-100
%end for loop
Data_iris_2 = [];
for i=1:10
new_index = randperm(size(D_iris2,1),size(D_iris2,1));
for j=1:size(D_iris2,1)
  index = new_index(j);
  D_iris2_temp(index,:) = D_iris2(j,:);
endfor
  Data_iris_2 = [Data_iris_2 D_iris2_temp];
  endfor
D_iris2_train = Data_iris_2(1:uint64(size(Data_iris_2,1).*0.75),:);
D_iris2_test = Data_iris_2(uint64(size(Data_iris_2,1).*0.75)+1:size(Data_iris_2,1),:);

Print2file_OHE(D_iris2_train,'D_iris3_test1  1.csv')%1- 100
Print2file_OHE(D_iris2_test,'D_iris3_test1.csv')%1-100
%end for loop

new_index = randperm(size(D_iris3,1),size(D_iris3,1));
for i=1:size(D_iris3,1)
  index = new_index(i);
  D_iris3(index,:) = D_iris3(i,:);
endfor
  
D_iris3_train = D_iris3(1:uint64(size(D_iris3,1).*0.75),:);
D_iris3_test = D_iris3(uint64(size(D_iris3,1).*0.75)+1:size(D_iris3,1),:);

Print2file_OHE(D_iris3_train,'D_iris3_train2.csv')%1- 100
Print2file_OHE(D_iris3_test,'D_iris3_test2.csv')%1-100
%end for loop