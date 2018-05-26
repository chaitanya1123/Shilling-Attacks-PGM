clear; clc; close all;

load('R_matrix.mat');
load('spam_users.mat');

idx = randperm(size(R,1));
R = R(idx,:);

datanum = size(R,1);
outputnum = 10;
inputnum = size(R,2);

inputdata = rand(datanum, inputnum);
outputdata = rand(datanum, outputnum);

rbm = randRBM(inputnum, outputnum);
rbm = pretrainRBM(rbm, R);

W = rbm.W;
b = rbm.b;
c = rbm.c;

for i = 1:1:size(R,1)
    x = R(i,:);
    free_energy(i) = -(c*x' + sum(log(exp(x*W + b) + 1)));
end

free_energy = free_energy';
spam = spam_users(idx);
predicted_spam = free_energy>-100;

true_pos = sum(spam == 1 & predicted_spam == 1);
pred_pos = sum(predicted_spam);
actual_pos = sum(spam);

precision = true_pos/pred_pos
recall = true_pos/actual_pos

spam_data = [spam free_energy];
idx_spam = find(spam_data(:,1) == 1);
spam_data = [idx_spam,spam_data(idx_spam,2)];

figure(1)
plot(free_energy,'o');
hold all;
scatter(spam_data(:,1),spam_data(:,2));
title('Spam User Detection');
ylabel('Free Energy');
xlabel('User ID');
legend('Non-Spam Users','Spam Users');
hold off;
