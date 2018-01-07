X = dlmread('wine.data.csv');
num_classes = 3;
n_train = 118;
n_test = 60;

%Extracting training and testing data from the input file
x_train = X(X(:, 1) == 1, 3:end);
m = mean(x_train, 1);
std_train = std(x_train);
%x_train = (x_train-m)./std_train;

train_labels = X(X(:, 1) == 1, 2);

x_test = X(X(:, 1) == 2, 3:end);
%x_test = (x_test-m)./std_train;
 
test_labels = X(X(:, 1) == 2, 2);

y_train = zeros(num_classes, n_train);
y_test = zeros(num_classes, n_test);

for i = 1:n_train
    y_train(train_labels(i), i) = 1; 
end

for i = 1:n_test
    y_test(test_labels(i), i) = 1; 
end

max_hidden_neurons = 10;
num_trials = 100;
errors = zeros(max_hidden_neurons, 1);

parfor i = 1:max_hidden_neurons
    acc = 0;
    for j = 1:num_trials
        %layers = 10*ones(1, i);
        %ANN.layers{1}.transferFcn = 'poslin';
        ANN.layers{1}.transferFcn = 'logsig';
        ANN = patternnet(i);
        [ANN,~] = train(ANN,x_train',y_train, 'useParallel', 'no');
        predictions = ANN(x_test');
        for a = 1:n_test
            predictions(:, a) = (predictions(:, a) == max(predictions(:, a)));
        end
        [error ,~] = confusion(predictions, y_test);
        acc = acc + error/num_trials;
    end
    errors(i) = acc;
end

ANN = patternnet(6);

[ANN,tr] = train(ANN,x_train',y_train);
predictions = ANN(x_test');
for i = 1:n_test
    predictions(:, i) = (predictions(:, i) == max(predictions(:, i)));
end
[c,cm] = confusion(predictions, y_test);

plot(errors);
[minErr, bestLayerSize] = min(errors);