clc;
clear;
close all;

% ///////////////////////////////////////////
%       Generate Data for SVM training     //
%////////////////////////////////////////////
% Generate features
n_examples = 100;
n_features = 2;
range = 4;
tol = 1e-2;
rand ('seed', n_features);
X_j = range*rand(n_examples, n_features);

% Define separation line
b = -6;
w = [4; -1];

% Get the label of data
y_j = sign(X_j*w + b);

% Visualize data
figure(1)
plot(X_j(y_j>0,1), X_j(y_j>0,2), 'o', 'MarkerFaceColor', [0.4660, 0.6740, 0.1880])
hold on
plot(X_j(y_j<0,1), X_j(y_j<0,2), 'o', 'MarkerFaceColor', [0.8660, 0.6740, 0.1880])

% Plot true separation line
x1 = 0; y1 = (-b -w(1)*x1)/w(2);
x2 = range; y2 = (-b -w(1)*x2)/w(2);
plot([x1, x2], [y1, y2], 'b', 'LineWidth', 1.5)
grid on
xlim([0, range])
ylim([0, range])
xlabel('x_1')
ylabel('x_2')
legend('+ve class', '-ve class', 'true plane')
title('Visualize data')

%///////////////////////////////////////////
%        Solve SVM Primal Problrm         //
%///////////////////////////////////////////
[weights, fval] = svm_primal_quadprog(X_j, y_j, n_features, n_examples);
w_est = weights(1:n_features);
b_est = weights(n_features+1);

figure(2)
plot(X_j(y_j>0,1), X_j(y_j>0,2), 'o', 'MarkerFaceColor', [0.4660, 0.6740, 0.1880])
hold on
plot(X_j(y_j<0,1), X_j(y_j<0,2), 'o', 'MarkerFaceColor', [0.8660, 0.6740, 0.1880])
grid on
xlim([0, range])
ylim([0, range])
xlabel('x_1')
ylabel('x_2')


% Plot w^TX + b = 0 plane
xest1 = 0; yest1 = (-b_est -w_est(1)*xest1)/w_est(2);
xest2 = range; yest2 = (-b_est -w_est(1)*xest2)/w_est(2);
plot([xest1, xest2], [yest1, yest2], 'r', 'LineWidth', 1)

% Plot w^TX + b = +1 plane
xest1 = 0; yest1 = (1-b_est -w_est(1)*xest1)/w_est(2);
xest2 = range; yest2 = (1-b_est -w_est(1)*xest2)/w_est(2);
plot([xest1, xest2], [yest1, yest2], '--r', 'LineWidth', 1)

% Plot w^TX + b = -1 plane
xest1 = 0; yest1 = (-1 -b_est -w_est(1)*xest1)/w_est(2);
xest2 = range; yest2 = (-1 -b_est -w_est(1)*xest2)/w_est(2);
plot([xest1, xest2], [yest1, yest2], '--r', 'LineWidth', 1)

title(strcat('SVM Primal Solution: Margin=', num2str(1/norm(w_est))));
legend('+ve class', '-ve class', 'primal SVM solution', 'primal SVM +ve margin', 'primal SVM -ve margin')

%///////////////////////////////////////////
%        Solve SVM Dual Problem           //
%///////////////////////////////////////////
[alpha, obj_val] = svm_dual_quadprog(X_j, y_j, n_examples);
support_vectors_index = find(alpha>tol);

%//////////////////////////////////////////////////////////////
%        Retrieve weights of the primal from dual's solution //
%//////////////////////////////////////////////////////////////
w_dual = (alpha.*y_j)'*X_j;
b_primal = y_j(support_vectors_index(3)) - w_dual*X_j(support_vectors_index(3),:)';

figure(3)
plot(X_j(y_j>0,1), X_j(y_j>0,2), 'o', 'MarkerFaceColor', [0.4660, 0.6740, 0.1880])
hold on
plot(X_j(y_j<0,1), X_j(y_j<0,2), 'o', 'MarkerFaceColor', [0.8660, 0.6740, 0.1880])
grid on
xlim([0, range])
ylim([0, range])
xlabel('x_1')
ylabel('x_2')

% Plot w^TX + b = 0 plane
x1 = 0; y1 = (-b_primal -w_dual(1)*x1)/w_dual(2);
x2 = range; y2 = (-b_primal -w_dual(1)*x2)/w_dual(2);
plot([x1, x2], [y1, y2], 'b', 'LineWidth', 1)

% Plot w^TX + b = +1 plane
x1 = 0; y1 = (1-b_primal -w_dual(1)*x1)/w_dual(2);
x2 = range; y2 = (1-b_primal -w_dual(1)*x2)/w_dual(2);
plot([x1, x2], [y1, y2], '--b', 'LineWidth', 1)

% Plot w^TX + b = -1 plane
x1 = 0; y1 = (-1-b_primal -w_dual(1)*x1)/w_dual(2);
x2 = range; y2 = (-1-b_primal -w_dual(1)*x2)/w_dual(2);
plot([x1, x2], [y1, y2], '--b', 'LineWidth', 1)

title(strcat('SVM Dual Solution: Margin=', num2str(1/norm(w_dual))));
legend('+ve class', '-ve class', 'dual SVM solution', 'dual SVM +ve margin', 'dual SVM -ve margin')