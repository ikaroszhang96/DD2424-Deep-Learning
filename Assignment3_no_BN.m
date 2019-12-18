% Use less data
%[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
%[valX, valY, valy] = LoadBatch('data_batch_2.mat');
%[testX, testY, testy] = LoadBatch('test_batch.mat');

% Use as much data
[Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat');
[Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat');
[Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat');
[Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat');
[Xtrain5,Ytrain5,ytrain5] = LoadBatch('data_batch_5.mat');
trainX=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
trainY=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
trainy=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];
valX = trainX(:, 1:5000);
valY = trainY(:, 1:5000);
valy = trainy(1:5000);
trainX = trainX(:, 5001:size(trainX, 2));
trainY = trainY(:, 5001:size(trainY, 2));
trainy = trainy(5001:size(trainy, 2));
[testX, testY, testy] = LoadBatch('test_batch.mat');

k = 3;
num_nodes = [50, 50];

d = size(trainX, 1);
K = size(trainY, 1);
m = [d, num_nodes, K];

std = 0.0001;
[W, b] = initWb(m,std);

GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 20;
GDparams.cur_batch = 0;
GDparams.eta_min = 1e-05;
GDparams.eta_max = 1e-01;
GDparams.n_s = 2250;

l_max = -3;
l_min = -5;
l = l_min + (l_max - l_min)*rand(1, 1);
%lambda = 10^l;
%lambda = 1.1988e-04;
%lambda = 2.1171e-05;
lambda = 0.005;

N = size(trainX, 2);
trainJ = zeros(1, GDparams.n_epochs + 1);
valJ = zeros(1, GDparams.n_epochs + 1);
for i = 1 : GDparams.n_epochs
    trainJ(i) = ComputeCost(trainX, trainY, W, b, lambda, k);
    valJ(i) = ComputeCost(valX, valY, W, b, lambda, k);
    for j = 1 : N / GDparams.n_batch
        j_start = (j - 1) * GDparams.n_batch + 1;
        j_end = j * GDparams.n_batch;
        inds = j_start : j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        GDparams.cur_batch = GDparams.cur_batch + 1;

        h = intervalues(Xbatch, W, b, k);
        P = EvaluateClassifier(h, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, h, P, W, lambda, k);

        if mod(floor(GDparams.cur_batch / GDparams.n_s), 2) == 0
            eta = GDparams.eta_min +(GDparams.cur_batch - floor(GDparams.cur_batch / GDparams.n_s) * GDparams.n_s) / GDparams.n_s * (GDparams.eta_max - GDparams.eta_min);
        end
        if mod(floor(GDparams.cur_batch / GDparams.n_s), 2) == 1
            eta = GDparams.eta_max - (GDparams.cur_batch - floor(GDparams.cur_batch / GDparams.n_s) * GDparams.n_s) / GDparams.n_s * (GDparams.eta_max - GDparams.eta_min);
        end
        for layer = 1 : k
            W{layer} = W{layer} - eta * grad_W{layer};
            b{layer} = b{layer} - eta * grad_b{layer};
        end
    end
    i
end

trainJ(GDparams.n_epochs + 1) = ComputeCost(trainX, trainY, W, b, lambda, k);
valJ(GDparams.n_epochs + 1) = ComputeCost(valX, valY, W, b, lambda, k);

lambda
train_accuracy = ComputeAccuracy(trainX, trainy, W, b, k);
disp("training accuracy:" + train_accuracy);
validation_accuracy = ComputeAccuracy(valX, valy, W, b, k);
disp("validation accuracy:" + validation_accuracy);
test_accuracy = ComputeAccuracy(testX, testy, W, b, k);
disp("test accuracy:" + test_accuracy);

% plot
figure()
plot(0 : GDparams.n_epochs, trainJ, 'r')
hold on
plot(0 : GDparams.n_epochs, valJ, 'b')
hold off
xlabel('epoch');
ylabel('loss');
legend('training loss', 'validation loss');

% Function for reading the data
function [X,Y,y] = LoadBatch(filename)
    addpath ./Datasets
    thename = load(filename);
    X = double(thename.data')/255.0;
    mean_X = mean(X, 2);
    std_X = std(X, 0, 2);
    X = X - repmat(mean_X, [1, size(X, 2)]);
    X = X ./ repmat(std_X, [1, size(X, 2)]);
    y = double(thename.labels');
    n = size(X,2);
    Y = zeros(10, n ,'double');
    for i = 1 : n
        Y( y(i)+1 ,i) = 1;
    end
end
               
function [W, b] = initWb(m, std)
    for i = 1 : size(m, 2) - 1
        if nargin < 2
            var = 2 / (m(i) + m(i + 1));
            std = sqrt(var);
        end

        W{i} = std * randn(m(i + 1), m(i));
        b{i} = zeros(m(i + 1), 1);
    end
end

% Function for evaluating network
function h = intervalues(X, W, b, k)
    eps = 0.001;
    for i = 1 : k - 1
        Wi = W{i};
        bi = b{i};
        bi = repmat(bi, 1, size(X, 2));
        s = Wi * X + bi;
        S{i} = s;
        X = max(0, s);
        h{i} = X;
    end
end

function P = EvaluateClassifier(h, W, b)
    W = W{end};
    b = b{end};
    X = h{end};
    b = repmat(b, 1, size(X, 2));
    s = W * X + b;
    denorm = repmat(sum(exp(s), 1), size(s, 1), 1);
    P = exp(s)./denorm;
end

% Function for computing cost function
function J = ComputeCost(X, Y, W, b, lambda, k)
    h = intervalues(X, W, b, k);
    P = EvaluateClassifier(h, W, b);
    P = log(P);
    n = size(X ,2);
    J1 = -sum(sum(P .* Y)) / n;
    J2 = 0;
    for i = 1 : length(W)
        J2 = J2 + lambda * sum(sum(W{i}.^2));
    end
    J = J1 + J2;
end

% Function for computing accuracy
function acc = ComputeAccuracy(X, y, W, b, k)
    h = intervalues(X, W, b, k);
    P = EvaluateClassifier(h, W, b);
    n = size(P, 2);
    acc = 0;
    max1 = 0;
    for i = 1 : n
        [max1, argmax] = max(P(:, i));
        if argmax == y(i) + 1
            acc = acc + 1;
        end
    end
    acc = acc / n;
end

% Function for computing gradients
function [grad_W, grad_b] = ComputeGradients(X, Y, h, P, W, lambda, k)
    grad_Wk = zeros(size(Y, 1), size(h{end}, 1));
    grad_bk = zeros(size(Y, 1), 1);
    g_prev = zeros(size(X, 2), size(h{end}, 1));
               
    n = size(X,2);
    G = - (Y - P);
    grad_Wk = G * h{k - 1}' / n;
    grad_bk = G * ones(n,1) / n;
    grad_Wk = 2 * lambda * W{k} + grad_Wk;
               
    grad_W{k} = grad_Wk;
    grad_b{k} = grad_bk;
               
    G = W{k}' * G;
    Ha = h{k - 1} > 0;
    G = G .* Ha;
               
    for l = k - 1 : -1 : 1
        if l == 1
            grad_W{l} = G * X';
        else
            grad_W{l} = G * h{l - 1}';
        end
        grad_W{l} = grad_W{l}/n + 2 * lambda * W{l};
        grad_b{l} = G * ones(n,1) / n;
        if l > 1
            G = W{l}' * G;
            G = G .* (h{l - 1} > 0);
        end
    end
end
