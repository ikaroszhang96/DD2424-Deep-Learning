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

k = 6;
%num_nodes = [50, 50];
%num_nodes = [1000, 1000];
%num_nodes = [50, 30, 20, 20, 10];
num_nodes = [1000, 500, 500, 500, 100];
%num_nodes = [50, 30, 20, 20, 10, 10, 10, 10];

d = size(trainX, 1);
K = size(trainY, 1);
m = [d, num_nodes, K];

std = 0.0001;
[W, b, Netparams] = initWb(m);
Netparams.W = W;
Netparams.b = b;
Netparams.use_bn = 1;
lambda = 0.005;

% Gradient Check
%[h, S, sbar, sw, mu, v] = intervalues1(trainX(:, 1:20), Netparams.W, Netparams.b, k, Netparams);
%P = EvaluateClassifier(h, Netparams.W, Netparams.b);
%[grad_W, grad_b, grad_gammas, grad_betas] = ComputeGradients1(trainX(:, 1:20), trainY(:, 1:20), h, S, sbar, P, Netparams.W, lambda, k, Netparams, mu, v);
%Grads = ComputeGradsNumSlow(trainX(:, 1:20), trainY(:, 1:20), Netparams, lambda, 1e-05, k);
%for ches = 1 : k
    %gradcheck_W{ches} = sum(abs(Grads.W{ches} - grad_W{ches})/max(1e-15, sum(abs(Grads.W{ches}) + abs(grad_W{ches}))));
%end
%for ches = 1 : k-1
    %gradcheck_ga{ches} = sum(abs(Grads.gammas{ches} - grad_gammas{ches})/max(1e-15, sum(abs(Grads.gammas{ches}) + abs(grad_gammas{ches}))));
%end
%for ches = 1 : k-1
    %gradcheck_be{ches} = sum(abs(Grads.betas{ches} - grad_betas{ches})/max(1e-15, sum(abs(Grads.betas{ches}) + abs(grad_betas{ches}))));
%end


GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 20;
GDparams.cur_batch = 0;
GDparams.eta_min = 1e-05;
GDparams.eta_max = 1e-01;
GDparams.n_s = 2250;

l_max = -2;
l_min = -3;
l = l_min + (l_max - l_min)*rand(1, 1);
%lambda = 10^l;
lambda = 0.005;

N = size(trainX, 2);
trainJ = zeros(1, GDparams.n_epochs + 1);
valJ = zeros(1, GDparams.n_epochs + 1);
for i = 1 : GDparams.n_epochs
    trainJ(i) = ComputeCost(trainX, trainY, W, b, lambda, k, Netparams);
    valJ(i) = ComputeCost(valX, valY, W, b, lambda, k, Netparams);

    colrank = randperm(size(trainX, 2));
    trainX = trainX(:, colrank);
    trainY = trainY(:, colrank);
    trainy = trainy(:, colrank);

    for j = 1 : N / GDparams.n_batch
        j_start = (j - 1) * GDparams.n_batch + 1;
        j_end = j * GDparams.n_batch;
        inds = j_start : j_end;
        Xbatch = trainX(:, inds);
        Ybatch = trainY(:, inds);
        GDparams.cur_batch = GDparams.cur_batch + 1;

        % Augment training data
        jitter = 0.01 * randn(size(Xbatch));
        Xbatch = Xbatch + jitter;


        [h, S, sbar, sw, mu, v] = intervalues(Xbatch, W, b, k, Netparams);
        P = EvaluateClassifier(h, W, b);

        for is = 1 : k - 1
            if j == 1
                mu_av = mu;
                v_av = v;
            else
                mu_av{is} = 0.5 * mu_av{is} + (1 - 0.5) * mu{is};
                v_av{is} = 0.5 * v_av{is} + (1 - 0.5) * v{is};
            end
        end

        [grad_W, grad_b, grad_gammas, grad_betas] = ComputeGradients(Xbatch, Ybatch, h, S, sbar, P, W, lambda, k, Netparams, mu_av, v_av);

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
        for layer = 1 : k - 1
            Netparams.gammas{layer} = Netparams.gammas{layer} - eta * grad_gammas{layer};
            Netparams.betas{layer} = Netparams.betas{layer} - eta * grad_betas{layer};
        end
    end
    i
end

trainJ(GDparams.n_epochs + 1) = ComputeCost(trainX, trainY, W, b, lambda, k, Netparams);
valJ(GDparams.n_epochs + 1) = ComputeCost(valX, valY, W, b, lambda, k, Netparams);

lambda
train_accuracy = ComputeAccuracy(trainX, trainy, W, b, k, Netparams);
disp("training accuracy:" + train_accuracy);
validation_accuracy = ComputeAccuracy(valX, valy, W, b, k, Netparams);
disp("validation accuracy:" + validation_accuracy);
test_accuracy = ComputeAccuracy(testX, testy, W, b, k, Netparams);
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
               
function [W, b, Netparams] = initWb(m, std)
    for i = 1 : size(m, 2) - 1
        if nargin < 2
            var = 2 / (m(i) + m(i + 1));
            std = sqrt(var);
        end

        W{i} = std * randn(m(i + 1), m(i));
        b{i} = zeros(m(i + 1), 1);
    end
    for i = 1 :size(m, 2) - 2
        Netparams.gammas{i} = randn(m(i + 1), 1);
        Netparams.betas{i} = randn(m(i + 1), 1);
    end
end

% Function for evaluating network
function [h, S, sbar, sw, mu, v] = intervalues(X, W, b, k, Netparams, mu_av, v_av)
    eps = 1e-15;
    for i = 1 : k - 1
        Wi = W{i};
        bi = b{i};
        bi = repmat(bi, 1, size(X, 2));
        s = Wi * X + bi;
        S{i} = s;
               
        if nargin < 7
            [sbar{i}, mui, vi] = BN_forward(s, eps);
        else
            [sbar{i}, mui, vi] = BN_forward(s, eps, mu_av{i}, v_av{i});
        end
        mu{i} = mui;
        v{i} = vi;
        sw{i} = Netparams.gammas{i} .* sbar{i} + Netparams.betas{i};

        X = max(0, sw{i});
        % Leaky Relu
        %X = max(0.1 * sw{i}, sw{i});
        h{i} = X;
    end
end

function [h, S, sbar, sw, mu, v] = intervalues1(X, W, b, k, Netparams, mu_av, v_av)
    eps = 1e-15;
    for i = 1 : k - 1
        Wi = W{i};
        bi = b{i};
        bi = repmat(bi, 1, size(X, 2));
        s = Wi * X + bi;
               
        X = max(0, s);
        % Leaky Relu
        %X = max(0.01 * s, s);
        S{i} = X;
               
        if nargin < 7
            [sbar{i}, mui, vi] = BN_forward(S{i}, eps);
        else
            [sbar{i}, mui, vi] = BN_forward(S{i}, eps, mu_av{i}, v_av{i});
        end
        mu{i} = mui;
        v{i} = vi;
        sw{i} = Netparams.gammas{i} .* sbar{i} + Netparams.betas{i};
               
        X = sw{i};
        h{i} = sw{i};
    end
end
               
function [sbar, mu, v] = BN_forward(s, eps, mu_av, v_av)
    if nargin < 4
        mu = mean(s, 2);
        v = mean((s - repmat(mu, 1, size(s, 2))).^2, 2);
    else
        mu = mu_av;
        v = v_av;
    end
    sbar = diag((v + eps).^(-0.5))*(s - repmat(mu, 1, size(s, 2)));
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
function J = ComputeCost(X, Y, W, b, lambda, k, Netparams)
    [h,~,~,~,~,~] = intervalues(X, W, b, k, Netparams);
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
function acc = ComputeAccuracy(X, y, W, b, k, Netparams)
    [h,~,~,~,~,~] = intervalues(X, W, b, k, Netparams);
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
function [grad_W, grad_b, grad_gammas, grad_betas] = ComputeGradients(X, Y, h, S, sbar, P, W, lambda, k, Netparams, mu, v)
    grad_Wk = zeros(size(Y, 1), size(h{end}, 1));
    grad_bk = zeros(size(Y, 1), 1);
    g_prev = zeros(size(X, 2), size(h{end}, 1));
    eps = 1e-15;
               
    n = size(X,2);
    G = - (Y - P);
    grad_Wk = G * h{k - 1}' / n;
    grad_bk = G * ones(n,1) / n;
    grad_Wk = 2 * lambda * W{k} + grad_Wk;
               
    grad_W{k} = grad_Wk;
    grad_b{k} = grad_bk;
               
    G = W{k}' * G;
               
    Ha = h{k - 1} > 0;
    % Leaky Relu
    %Ha = ((h{k - 1} > 0) + 0.1 * (h{k - 1} < 0));
    G = G .* Ha;
               
    for l = k - 1 : -1 : 1
        grad_gammas{l} = (G .* sbar{l}) * ones(n,1) / n;
        grad_betas{l} = G * ones(n,1) / n;
        G = G .* (Netparams.gammas{l} * ones(1,n));
        G = BN_backward(G, S{l}, mu{l}, v{l}, eps);
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
            % Leaky Relu
            %G = G .* ((h{l - 1} > 0) + 0.1 * (h{l - 1} < 0));
        end
    end
end

function [grad_W, grad_b, grad_gammas, grad_betas] = ComputeGradients1(X, Y, h, S, sbar, P, W, lambda, k, Netparams, mu, v)
    grad_Wk = zeros(size(Y, 1), size(h{end}, 1));
    grad_bk = zeros(size(Y, 1), 1);
    g_prev = zeros(size(X, 2), size(h{end}, 1));
    eps = 1e-15;
    
    n = size(X,2);
    G = - (Y - P);
    grad_Wk = G * h{k - 1}' / n;
    grad_bk = G * ones(n,1) / n;
    grad_Wk = 2 * lambda * W{k} + grad_Wk;
               
    grad_W{k} = grad_Wk;
    grad_b{k} = grad_bk;
               
    G = W{k}' * G;
               
    grad_gammas{k - 1} = (G .* sbar{k - 1}) * ones(n,1) / n;
    grad_betas{k - 1} = G * ones(n,1) / n;
    G = G .* (Netparams.gammas{k - 1} * ones(1,n));
    G = BN_backward(G, S{k - 1}, mu{k - 1}, v{k - 1}, eps);
               
    for l = k - 1 : -1 : 1
               
        G = G .* (S{l} > 0);
               
        if l == 1
            grad_W{l} = G * X';
        else
            grad_W{l} = G * h{l - 1}';
        end
        grad_W{l} = grad_W{l}/n + 2 * lambda * W{l};
        grad_b{l} = G * ones(n,1) / n;
        if l > 1
            G = W{l}' * G;
            
            grad_gammas{l - 1} = (G .* sbar{l - 1}) * ones(n,1) / n;
            grad_betas{l - 1} = G * ones(n,1) / n;
            G = G .* (Netparams.gammas{l - 1} * ones(1,n));
            G = BN_backward(G, S{l - 1}, mu{l - 1}, v{l - 1}, eps);
        end
    end
end

function Gb = BN_backward(G, S, mu, v, eps)
    sigma1 = (v + eps) .^ -0.5;
    sigma2 = (v + eps) .^ -1.5;
    n = size(G,2);
    G1 = G .* (sigma1 * ones(1,n));
    G2 = G .* (sigma2 * ones(1,n));
    D = S - mu * ones(1,n);
    c = (G2 .* D) * ones(n,1);
    Gb = G1 - (G1 * ones(n,1) / n) - (D .* (c * ones(1,n)) / n);
end
               
function Grads = ComputeGradsNumSlow(X, Y, Netparams, lambda, h, k)
               
               
    Grads.W = cell(numel(Netparams.W), 1);
    Grads.b = cell(numel(Netparams.b), 1);
    if Netparams.use_bn
        Grads.gammas = cell(numel(Netparams.gammas), 1);
        Grads.betas = cell(numel(Netparams.betas), 1);
    end
               
    for j=1:length(Netparams.b)
        Grads.b{j} = zeros(size(Netparams.b{j}));
        NetTry = Netparams;
        for i=1:length(Netparams.b{j})
            b_try = Netparams.b;
            b_try{j}(i) = b_try{j}(i) - h;
            NetTry.b = b_try;
            c1 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
            b_try = Netparams.b;
            b_try{j}(i) = b_try{j}(i) + h;
            NetTry.b = b_try;
            c2 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
            Grads.b{j}(i) = (c2-c1) / (2*h);
        end
    end
               
    for j=1:length(Netparams.W)
        Grads.W{j} = zeros(size(Netparams.W{j}));
        NetTry = Netparams;
        for i=1:numel(Netparams.W{j})
               
            W_try = Netparams.W;
            W_try{j}(i) = W_try{j}(i) - h;
            NetTry.W = W_try;
            c1 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
            W_try = Netparams.W;
            W_try{j}(i) = W_try{j}(i) + h;
            NetTry.W = W_try;
            c2 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
            Grads.W{j}(i) = (c2-c1) / (2*h);
        end
    end
               
    if Netparams.use_bn
        for j=1:length(Netparams.gammas)
            Grads.gammas{j} = zeros(size(Netparams.gammas{j}));
            NetTry = Netparams;
            for i=1:numel(Netparams.gammas{j})
               
               gammas_try = Netparams.gammas;
               gammas_try{j}(i) = gammas_try{j}(i) - h;
               NetTry.gammas = gammas_try;
               c1 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
               gammas_try = Netparams.gammas;
               gammas_try{j}(i) = gammas_try{j}(i) + h;
               NetTry.gammas = gammas_try;
               c2 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
               Grads.gammas{j}(i) = (c2-c1) / (2*h);
            end
        end
               
        for j=1:length(Netparams.betas)
            Grads.betas{j} = zeros(size(Netparams.betas{j}));
            NetTry = Netparams;
            for i=1:numel(Netparams.betas{j})
               
               betas_try = Netparams.betas;
               betas_try{j}(i) = betas_try{j}(i) - h;
               NetTry.betas = betas_try;
               c1 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
               betas_try = Netparams.betas;
               betas_try{j}(i) = betas_try{j}(i) + h;
               NetTry.betas = betas_try;
               c2 = ComputeCost(X, Y, NetTry.W , NetTry.b, lambda, k, NetTry);
               
               Grads.betas{j}(i) = (c2-c1) / (2*h);
            end
        end
    end
end
