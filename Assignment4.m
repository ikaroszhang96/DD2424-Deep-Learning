% Read data
book_fname = 'data/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);
book_chars = unique(book_data);

char_to_ind = containers.Map('KeyType', 'char', 'ValueType', 'int32');
ind_to_char = containers.Map('KeyType', 'int32', 'ValueType', 'char');

keySet = num2cell(book_chars);
valueSet = 1 : length(keySet);
newMap1 = containers.Map(keySet, valueSet);
newMap2 = containers.Map(valueSet, keySet);
char_to_ind = [char_to_ind; newMap1];
ind_to_char = [ind_to_char; newMap2];

% Initialization
K = length(keySet);
m = 100;
eta = 0.1;
seq_length = 25;
sig = 0.01;
RNN.b = zeros(m, 1);
RNN.c = zeros(K, 1);
RNN.U = randn(m, K)*sig;
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;

Xi = zeros(1, length(book_data));
for i = 1 : length(book_data)
    Xi(i) = char_to_ind(book_data(i));
end
X = oneHot(Xi, K);
Xcheck = X(:, 1 : seq_length);
Ycheck = X(:, 2 : seq_length + 1);

% Gradcheck
h0 = zeros(size(RNN.W, 1), 1);
[~, a, h, ~, p] = forward_Pass(RNN, Xcheck, Ycheck, h0, 25, K, m);
grads = ComputeGradients(RNN, Xcheck, Ycheck, a, h, p, 25, m);
n_grads = ComputeGradsNum(Xcheck, Ycheck, RNN, 1e-04);
eps = 1e-15;
for f = fieldnames(RNN)'
    truegrad = n_grads.(f{1});
    compgrad = grads.(f{1});
    gradcheck = max(abs(truegrad - compgrad))/max(eps, sum(abs(truegrad) + abs(compgrad)));
    disp(f{1});
    disp(num2str(gradcheck));
end

% Train
Xi = zeros(1, length(book_data));
for i = 1 : length(book_data)
    Xi(i) = char_to_ind(book_data(i));
end
X = oneHot(Xi, K);
Y = X;
iter = 1;
n_epochs = 10;
SL = [];
sl = 0;
hprev = [];
min_loss = 1000000;

% Adagrad parameters
M.b = zeros(size(RNN.b));
M.c = zeros(size(RNN.c));
M.U = zeros(size(RNN.U));
M.W = zeros(size(RNN.W));
M.V = zeros(size(RNN.V));

for i = 1 : n_epochs
    e = 1;
    len = 200;
    smooth_loss = sl(end);
    sl = [];
    while e <= length(X) - seq_length - 1
        Xe = X(:, e : e + seq_length - 1);
        Ye = Y(:, e + 1 : e + seq_length);
        if e == 1
            hprev = zeros(m, 1);
        else
            hprev = h(:, end);
        end

        [loss, a, h, ~, p] = forward_Pass(RNN, Xe, Ye, hprev, seq_length, K, m);
        grads = ComputeGradients(RNN, Xe, Ye, a, h, p, seq_length, m);

        % Adagrad
        eps = 1e-10;
        for f = fieldnames(RNN)'
            %grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
            M.(f{1}) = M.(f{1}) + grads.(f{1}).^2;
            RNN.(f{1}) = RNN.(f{1}) -  eta * (grads.(f{1}) ./ (M.(f{1}) + eps).^(0.5));
        end

        if iter == 1 && e == 1
            smooth_loss = loss;
        end
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss;

        if smooth_loss < min_loss
            best_RNN = RNN;
            best_h = hprev;
            best_iter = iter;
            best_loss = smooth_loss;
        end
        sl = [sl, smooth_loss];

        if iter == 1 || mod(iter, 10000) == 0
            y = synthesize(RNN, hprev, X(:, 1), len, K);
            c = [];
            for i = 1 : len
                c = [c ind_to_char(y(i))];
            end
            disp('--------');
            disp(['iter = ' num2str(iter) ', smooth_loss = ' num2str(smooth_loss)]);
            disp(c);
        end

        iter = iter + 1;
        e = e + seq_length;
    end
    SL = [SL, sl];
end

y = synthesize(best_RNN, best_h, X(:, 1), 1000, K);
c = [];
for i = 1 : 1000
    c = [c ind_to_char(y(i))];
end
disp('--------');
disp(['iter = ' num2str(best_iter) ', smooth_loss = ' num2str(best_loss)]);
disp(c);

figure()
plot(1 : length(SL), SL, 'r')
xlabel('Iteration step');
ylabel('smooth_loss');

% Function for synthesizing the text
function y = synthesize(RNN, h0, x0, n, K)
    W = RNN.W;
    U = RNN.U;
    V = RNN.V;
    b = RNN.b;
    c = RNN.c;
    h = h0;
    x = x0;
    y = zeros(1, n);

    for t = 1 : n
        a = W * h + U * x + b;
        h = tanh(a);
        o = V * h + c;
        p = exp(o);
        p = p / sum(p);

        cp = cumsum(p);
        a = rand;
        ixs = find(cp - a > 0);
        ii = ixs(1);

        x = oneHot(ii, K);
        y(t) = ii;
    end
end

% Create one-hot expression
function one = oneHot(label, K)

    N = length(label);
    one = zeros(K, N);
    for i = 1 : N
        one(label(i), i) = 1;
    end
end

% Gradient function
function grads = ComputeGradients(RNN, X, Y, a, h, p, n, m)

    W = RNN.W;
    V = RNN.V;
    dh = zeros(n, m);
    da = zeros(n, m);

    g = -(Y - p)';
    grads.c = (sum(g))';
    grads.V = g' * h(:, 2 : end)';

    dh(n, :) = g(n, :) * V;
    da(n, :) = dh(n, :) * diag(1 - (tanh(a(:, n))).^2);

    for t = n - 1 : -1 : 1
        dh(t, :) = g(t, :) * V + da(t + 1, :) * W;
        da(t, :) = dh(t, :) * diag(1 - (tanh(a(:, t))).^2);
    end

    grads.b = (sum(da))';
    grads.W = da' * h(:, 1 : end - 1)';
    grads.U = da' * X';

end

% Loss function
function loss = ComputeLoss(X, Y, RNN, h)
    W = RNN.W;
    U = RNN.U;
    V = RNN.V;
    b = RNN.b;
    c = RNN.c;
    n = size(X, 2);
    loss = 0;

    for t = 1 : n
        at = W * h + U * X(:, t) + b;
        h = tanh(at);
        o = V * h + c;
        pt = exp(o);
        p = pt / sum(pt);

        loss = loss - log(Y(:, t)'*p);
    end
end

% Forward pass of RNN
function [loss, a, h, o, p] = forward_Pass(RNN, X, Y, h0, n, K, m)
    W = RNN.W;
    U = RNN.U;
    V = RNN.V;
    b = RNN.b;
    c = RNN.c;
    ht = h0;
    o = zeros(K, n);
    p = zeros(K, n);
    h = zeros(m, n);
    a = zeros(m, n);
    loss = 0;

    for t = 1 : n
        at = W * ht + U * X(:, t) + b;
        a(:, t) = at;
        ht = tanh(at);
        h(:, t) = ht;
        o(:, t) = V * ht + c;
        pt = exp(o(:, t));
        p(:, t) = pt/sum(pt);

        loss = loss - log(Y(:, t)'*p(:, t));
    end
    h = [h0, h];
end

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, h);
    end
end
                          
function grad = ComputeGradNumSlow(X, Y, f, RNN, h)
                          
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end
end
