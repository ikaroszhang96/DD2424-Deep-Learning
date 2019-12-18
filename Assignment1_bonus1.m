% Exercise 1.1 loading data
[initX,initY,inity] = LoadBatch('data_batch_1.mat');

% Exercise 1.2 initializing parameters
K = size(initY, 1);
d = size(initX, 1);

% Xavier initialization
W = normrnd(0, 1/sqrt(d), [K, d]);
b = normrnd(0, 0.01, [K, 1]);

%[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
%[valX, valY, valy] = LoadBatch('data_batch_2.mat');
%[testX, testY, testy] = LoadBatch('test_batch.mat');

% Use all data
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

lambda = 0;

% Exerxise 1.3 evaluate the network
P = EvaluateClassifier(trainX(:, 1:100), W, b);

% Exercise 1.6 Check Gradient
[grad_W, grad_b] = ComputeGradients(trainX(:, 1:100), trainY(:, 1:100), P, W, lambda);
[grad_b1, grad_W1] = ComputeGradsNumSlow(trainX(:, 1:100), trainY(:, 1:100), W, b, lambda, 1e-6);

W_error = abs(grad_W - grad_W1);
denomW = max(1e-15, abs(grad_W) + abs(grad_W1));
error_W = sum(sum(W_error ./ denomW)) / numel(grad_W);
disp("W_error:" + error_W);

b_error = abs(grad_b - grad_b1);
denomb = max(1e-15, abs(grad_b) + abs(grad_b1));
error_b = sum(sum(b_error ./ denomb)) / numel(grad_b);
disp("b_error:" + error_b);


% Exercise 1.7 Mini batch GD
GDparams.n_batch = 25;
GDparams.eta = 0.001;
GDparams.n_epochs = 100;

trainJ = zeros(1, GDparams.n_epochs);
valJ = zeros(1, GDparams.n_epochs);
for i = 1 : GDparams.n_epochs
    trainJ(i) = ComputeCost(trainX, trainY, W, b, lambda);
    valJ(i) = ComputeCost(valX, valY, W, b, lambda);
    [W, b] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda);
    i
    % Learning Rate Decay
    if mod(i, 10) == 0
        GDparams.eta = GDparams.eta * 0.99;
    end
end

% Exerxise 1.5 compute the accuracy of the network
train_accuracy = ComputeAccuracy(trainX, trainy, W, b);
disp("training accuracy:" + train_accuracy);
test_accuracy = ComputeAccuracy(testX, testy, W, b);
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
function P = EvaluateClassifier(X,W,b)
    % P = W * X + b
    P = W * X;
    for i = 1 : 10
        P(i, :) = P(i, :) + b(i);
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
    P = EvaluateClassifier(X ,W ,b);
    P = log(P);
    n = size(X ,2);
    J = -sum(sum(P .* Y)) / n + lambda * sum(sum(W .* W));
end

% Function for computing accuracy
function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
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
function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    G = -(Y - P)';
    grad_W = G' * X';
    grad_b = sum(G, 1)';
    n = size(X, 2);
    grad_W = grad_W / n + 2 * lambda * W;
    grad_b = grad_b / n;
end
               
% Function used to check gradients
function [grad_b, grad_W] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
               
    no = size(W, 1);
    d = size(X, 1);
               
    grad_W = zeros(size(W));
    grad_b = zeros(no, 1);
               
    for i=1:length(b)
        b_try = b;
        b_try(i) = b_try(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        b_try = b;
        b_try(i) = b_try(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        grad_b(i) = (c2-c1) / (2*h);
    end
               
    for i=1:numel(W)
               
        W_try = W;
        W_try(i) = W_try(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
               
        W_try = W;
        W_try(i) = W_try(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
               
        grad_W(i) = (c2-c1) / (2*h);
    end
end
               

% Function used to perform minibatch GD
function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)
    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    N = size(X, 2);

    for j = 1 : N/n_batch
        j_start = (j - 1) * n_batch + 1;
        j_end = j * n_batch;
        inds = j_start : j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);
               
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
               
        W = W - eta * grad_W;
        b = b - eta * grad_b;
    end
               
    Wstar = W;
    bstar = b;
end
