%[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
%[valX, valY, valy] = LoadBatch('data_batch_2.mat');
%[testX, testY, testy] = LoadBatch('test_batch.mat');
lambda = 0.01;

% Use as much data
[Xtrain1,Ytrain1,ytrain1] = LoadBatch('data_batch_1.mat');
[Xtrain2,Ytrain2,ytrain2] = LoadBatch('data_batch_2.mat');
[Xtrain3,Ytrain3,ytrain3] = LoadBatch('data_batch_3.mat');
[Xtrain4,Ytrain4,ytrain4] = LoadBatch('data_batch_4.mat');
[Xtrain5,Ytrain5,ytrain5] = LoadBatch('data_batch_5.mat');
trainX=[Xtrain1,Xtrain2,Xtrain3,Xtrain4,Xtrain5];
trainY=[Ytrain1,Ytrain2,Ytrain3,Ytrain4,Ytrain5];
trainy=[ytrain1,ytrain2,ytrain3,ytrain4,ytrain5];
valX = trainX(:, 1:1000);
valY = trainY(:, 1:1000);
valy = trainy(1:1000);
trainX = trainX(:, 1001:size(trainX, 2));
trainY = trainY(:, 1001:size(trainY, 2));
trainy = trainy(1001:size(trainy, 2));
[testX, testY, testy] = LoadBatch('test_batch.mat');

 
m = 500;
d = size(trainX, 1);
K = size(trainY, 1);

% Xavier
%W1 = 1/sqrt(d) * randn(m, d);
%W2 = 1/sqrt(m) * randn(K, m);

% He
W1 = sqrt(2/d) * randn(m, d);
W2 = sqrt(2/m) * randn(K, m);

b1 = zeros(m, 1);
b2 = zeros(K, 1);

W = {W1, W2};
b = {b1, b2};

% Check Gradient
h = EvaluateClassifier1(trainX(:, 1:100), W, b);
P = EvaluateClassifier2(h, W, b);
[grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(trainX(:, 1:100), trainY(:, 1:100), h, P, W, b, lambda);
 %[tgrad_b, tgrad_W] = ComputeGradsNumSlow(trainX(:, 1:100), trainY(:, 1:100), W, b, lambda, 1e-5);
 %eps = 1e-15;
 %gradcheck_b1 = sum(abs(tgrad_b{1} - grad_b1)/max(eps, sum(abs(tgrad_b{1}) + abs(grad_b1))))
 %gradcheck_W1 = sum(sum(abs(tgrad_W{1} - grad_W1)/max(eps, sum(sum(abs(tgrad_W{1}) + abs(grad_W1))))))
 %gradcheck_b2 = sum(abs(tgrad_b{2} - grad_b2)/max(eps, sum(abs(tgrad_b{2}) + abs(grad_b2))))
 %gradcheck_W2 = sum(sum(abs(tgrad_W{2} - grad_W2)/max(eps, sum(sum(abs(tgrad_W{2}) + abs(grad_W2))))))

GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;
GDparams.cur_batch = 0;
%GDparams.eta_min = 1e-05;
GDparams.eta_min = 1e-05;
GDparams.eta_max = 1e-01;
%GDparams.eta_max = 0.05;
GDparams.n_s = 490;

l_max = -3;
l_min = -5;
l = l_min + (l_max - l_min)*rand(1, 1);
%lambda = 10^l;
lambda = 0.005;
%lambda = 1.1988e-04;
%lambda = 2.1171e-05;

N = size(trainX, 2);
trainJ = zeros(1, GDparams.n_epochs + 1);
valJ = zeros(1, GDparams.n_epochs + 1);
%trainA = zeros(1, GDparams.n_epochs + 1);
testA = zeros(1, 4);
etas = zeros(1, 4);
%eta = GDparams.eta_min;
k = 1;
for i = 1 : GDparams.n_epochs
    %etas(i) = eta;
    %trainA(i) = ComputeAccuracy(trainX, trainy, W, b);
    %valA(i) = ComputeAccuracy(valX, valy, W, b);
    trainJ(i) = ComputeCost(trainX, trainY, W, b, lambda);
    valJ(i) = ComputeCost(valX, valY, W, b, lambda);
    for j = 1 : N / GDparams.n_batch
        j_start = (j - 1) * GDparams.n_batch + 1;
        j_end = j * GDparams.n_batch;
        inds = j_start : j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        GDparams.cur_batch = GDparams.cur_batch + 1;
        % Augment training data
        jitter = 0.005 * randn(size(Xbatch));
        Xbatch = Xbatch + jitter;
 
        h = EvaluateClassifier1(Xbatch, W, b);
        P = EvaluateClassifier2(h, W, b);
        [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(Xbatch, Ybatch, h, P, W, b, lambda);
 
        if mod(floor(GDparams.cur_batch / GDparams.n_s), 2) == 0
            eta = GDparams.eta_min +(GDparams.cur_batch - floor(GDparams.cur_batch / GDparams.n_s) * GDparams.n_s) / GDparams.n_s * (GDparams.eta_max - GDparams.eta_min);
        end
        if mod(floor(GDparams.cur_batch / GDparams.n_s), 2) == 1
            eta = GDparams.eta_max - (GDparams.cur_batch - floor(GDparams.cur_batch / GDparams.n_s) * GDparams.n_s) / GDparams.n_s * (GDparams.eta_max - GDparams.eta_min);
        end
        W{1} = W{1} - eta * grad_W1;
        b{1} = b{1} - eta * grad_b1;
        W{2} = W{2} - eta * grad_W2;
        b{2} = b{2} - eta * grad_b2;
        if mod(j,98) == 0
            etas(k) = eta;
            testA(k) = ComputeAccuracy(testX, testy, W, b);
            k = k + 1;
        end
    end
    i
end
%etas(GDparams.n_epochs + 1) = eta;
%trainA(GDparams.n_epochs + 1) = ComputeAccuracy(trainX, trainy, W, b);
%valA(GDparams.n_epochs + 1) = ComputeAccuracy(valX, valy, W, b);
trainJ(GDparams.n_epochs + 1) = ComputeCost(trainX, trainY, W, b, lambda);
valJ(GDparams.n_epochs + 1) = ComputeCost(valX, valY, W, b, lambda);

lambda
train_accuracy = ComputeAccuracy(trainX, trainy, W, b);
disp("training accuracy:" + train_accuracy);
validation_accuracy = ComputeAccuracy(valX, valy, W, b);
disp("validation accuracy:" + validation_accuracy);
test_accuracy = ComputeAccuracy(testX, testy, W, b);
disp("test accuracy:" + test_accuracy);

% plot
figure()
plot(0 : GDparams.n_epochs, trainJ, 'r')
hold on
plot(0 : GDparams.n_epochs, valJ, 'b')
hold off
xlabel('epoch');
ylabel('cost');
legend('training cost', 'validation cost');
 
% plot
figure()
plot(etas, testA, 'r')
xlabel('eta');
ylabel('accuracy');

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
               
% Function for evaluating network
function P = EvaluateClassifier1(X, W, b)
    W1 = cell2mat(W(1));
    b1 = cell2mat(b(1));
    P = W1 * X;
    for i = 1 : size(b1, 1)
        P(i, :) = P(i, :) + b1(i);
    end
    P = max(0, P);
end
               
function P = EvaluateClassifier2(X, W, b)
    W2 = cell2mat(W(2));
    b2 = cell2mat(b(2));
    P = W2 * X;
    for i = 1 : size(b2, 1)
        P(i, :) = P(i, :) + b2(i);
    end
               
    % Softmax
    P = exp(P);
    n = size(P, 2);
    for i = 1 : n
        P(:, i) = P(:, i) / sum(P(:, i));
    end
end

% Function for computing cost function
function J = ComputeCost(X, Y, W, b, lambda)
    W1 = cell2mat(W(1));
    W2 = cell2mat(W(2));
    h = EvaluateClassifier1(X, W, b);
    P = EvaluateClassifier2(h, W, b);
    P = log(P);
    n = size(X ,2);
    J = -sum(sum(P .* Y)) / n + lambda * (sum(sum(W1 .* W1)) + sum(sum(W2 .* W2)));
end

% Function for computing accuracy
function acc = ComputeAccuracy(X, y, W, b)
    h = EvaluateClassifier1(X, W, b);
    P = EvaluateClassifier2(h, W, b);
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
function [grad_W1, grad_b1, grad_W2, grad_b2] = ComputeGradients(X, Y, P1, P2, W, b, lambda)
    W1 = W{1};
    W2 = W{2};
    b1 = b{1};
    b2 = b{2};
    grad_W1 = zeros(size(W1));
    grad_W2 = zeros(size(W2));
    grad_b1 = zeros(size(b1));
    grad_b2 = zeros(size(b2));
    
    n = size(X,2);
    G = - (Y - P2);
    grad_W2 = G * P1' / n;
    grad_b2 = G * ones(n,1) / n;
               
    G = W2' * G;
    Ha = P1 > 0;
    G = G .* Ha;
    grad_W1 = G * X' / n;
    grad_b1 = G * ones(n,1) / n;
    
    grad_W1 = 2*lambda*W1 + grad_W1;
    grad_W2 = 2*lambda*W2 + grad_W2;
end

% Function used to check gradients
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
                                                               
    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
                                                               
    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
                                                               
        for i=1:length(b{j})
                                                               
            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);
                                                               
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
                                                               
            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end
                                                       
    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));
                                                               
        for i=1:numel(W{j})
                                                               
            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);
                                                               
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);
                                                               
            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end
