% Exercise 1.1 loading data
[initX,initY,inity] = LoadBatch('data_batch_1.mat');

% Exercise 1.2 initializing parameters
K = size(initY, 1);
d = size(initX, 1);

% Xavier initialization
W = normrnd(0, 0.01, [K, d]);
b = normrnd(0, 0.01, [K, 1]);

[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[valX, valY, valy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

% Use all data
%[Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat');
%[Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat');
%[Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat');
%[Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat');
%[Xtrain5,Ytrain5,ytrain5] = LoadBatch('data_batch_5.mat');
%trainX=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
%trainY=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
%trainy=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];
%valX = trainX(:, 1:1000);
%valY = trainY(:, 1:1000);
%valy = trainy(1:1000);
%trainX = trainX(:, 1001:size(trainX, 2));
%trainY = trainY(:, 1001:size(trainY, 2));
%trainy = trainy(1001:size(trainy, 2));
%[testX, testY, testy] = LoadBatch('test_batch.mat');

lambda = 0.1;

% Exerxise 1.3 evaluate the network
S = EvaluateClassifierSVM(trainX(:, 1:100), W, b);

% Exercise 1.6 Check Gradient
[grad_W, grad_b] = ComputeGradientsSVM(trainX(:, 1:100), trainy(:, 1:100), S, W, lambda);
[grad_b1, grad_W1] = ComputeGradsNumSlowSVM(trainX(:, 1:100), trainy(:, 1:100), W, b, lambda, 1e-6);

W_error = abs(grad_W - grad_W1);
denomW = max(1e-15, abs(grad_W) + abs(grad_W1));
error_W = sum(sum(W_error ./ denomW)) / numel(grad_W);
disp("W_error:" + error_W);

b_error = abs(grad_b - grad_b1);
denomb = max(1e-15, abs(grad_b) + abs(grad_b1));
error_b = sum(sum(b_error ./ denomb)) / numel(grad_b);
disp("b_error:" + error_b);

% Exercise 1.7 Mini batch GD
GDparams.n_batch = 100;
GDparams.eta = 0.001;
GDparams.n_epochs = 40;

trainJ = zeros(1, GDparams.n_epochs);
valJ = zeros(1, GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
    trainJ(i) = ComputeCostSVM(trainX, trainy, W, b, lambda);
    valJ(i) = ComputeCostSVM(valX, valy, W, b, lambda);
    [W, b] = MiniBatchGDSVM(trainX, trainy, GDparams, W, b, lambda);
    i
end

% Exerxise 1.5 compute the accuracy of the network
train_accuracy = ComputeAccuracySVM(trainX, trainy, W, b);
disp("training accuracy:" + train_accuracy);
test_accuracy = ComputeAccuracySVM(testX, testy, W, b);
disp("test accuracy:" + test_accuracy);

% plot
figure()
plot(1 : GDparams.n_epochs, trainJ, 'r')
hold on
plot(1 : GDparams.n_epochs, valJ, 'b')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');

% Visualization
for i = 1 : K
im = reshape(W(i, :), 32, 32, 3);
s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure()
montage(s_im, 'size', [1, K])

% Function for reading the data
function [X,Y,y] = LoadBatch(filename)
    addpath ./Datasets
    thename = load(filename);
    X = double(thename.data')/255.0;
    y = double(thename.labels');
    n = size(X,2);
    Y = zeros(10, n ,'double');
    for i = 1 : n
        Y( y(i)+1 ,i) = 1;
    end
end
               
% Function for evaluating network
               
function S = EvaluateClassifierSVM(X, W, b)
    S = W * X;
    for i = 1 : 10
        S(i, :) = S(i, :) + b(i);
    end
end

% Function for computing cost function
               
function J = ComputeCostSVM(X, y, W, b, lambda)
    S = EvaluateClassifierSVM(X, W, b);
    n = size(X, 2);
    J = 0;
    for i = 1 : n
        S(:, i) = S(:, i) - S(y(i) + 1, i) + 1;
        S(:, i) = max(0, S(:, i));
        J = J - 1 + sum(S(:, i));
    end
    J = J / n + lambda * sum(sum(W .* W));
end

% Function for computing accuracy

function acc = ComputeAccuracySVM(X, y, W, b)
    S = EvaluateClassifierSVM(X, W, b);
    n = size(S, 2);
    acc = 0;
    max1 = 0;
    for i = 1 : n
        [max1, argmax] = max(S(:, i));
        if argmax == y(i) + 1
            acc = acc + 1;
        end
    end
    acc = acc / n;
end
               
% Function for computing gradients

function [grad_W, grad_b] = ComputeGradientsSVM(X, y, S, W, lambda)
    n = size(X, 2);
    d = size(X, 1);
    K = size(S, 1);
    grad_W = zeros(K, d);
    grad_b = zeros(K, 1);
    for i = 1 : n
        %s_j - s_y + 1
        S(:, i) = S(:, i) - S(y(i) + 1, i) + 1;
        % indicator function
        S(:, i) = S(:, i) > 0;
        % -sum of indicator for j=y
        S(y(i) + 1, i) = S(y(i) + 1, i) - sum(S(:, i));
        grad_W = grad_W + S(:, i) * X(:, i)';
        % equivalent to X(:, i) = 1 for grad_b
        grad_b = grad_b + S(:, i);
    end
    grad_W = grad_W / n + 2 * lambda * W;
    grad_b = grad_b / n;
end
               
% Function used to check gradients
               
function [grad_b, grad_W] = ComputeGradsNumSlowSVM(X, y, W, b, lambda, h)
               
    no = size(W, 1);
    d = size(X, 1);
               
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
               
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCostSVM(X, y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCostSVM(X, y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end
               
    for i=1:numel(W)
               
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCostSVM(X, y, W_try, b, lambda);
               
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCostSVM(X, y, W_try, b, lambda);
               
        grad_W(i) = (c2-c1) / (2*h);
    end
end


% Function used to perform minibatch GD
function [Wstar, bstar] = MiniBatchGDSVM(X, Y, GDparams, W, b, lambda)
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    N = size(X, 2);
               
    for j = 1 : N/n_batch
        j_start = (j - 1) * n_batch + 1;
        j_end = j * n_batch;
        inds = j_start : j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
               
        S = EvaluateClassifierSVM(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradientsSVM(Xbatch, Ybatch, S, W, lambda);
               
        W = W - eta * grad_W;
        b = b - eta * grad_b;
    end
               
    Wstar = W;
    bstar = b;
end
