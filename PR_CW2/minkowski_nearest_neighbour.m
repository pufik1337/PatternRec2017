X = dlmread('wine.data.csv');
num_classes = 3;
n_train = 118;
n_test = 60;
%K = 1;

%Extracting training and testing data from the input file
x_train = X(X(:, 1) == 1, 3:end);
m = mean(x_train, 1);
std_train = std(x_train);

%Normalizing the training data
x_train = (x_train-m)./std_train;

train_labels = X(X(:, 1) == 1, 2);

x_test = X(X(:, 1) == 2, 3:end);

%Normalizing the test data
x_test = (x_test-m)./std_train;
 
test_labels = X(X(:, 1) == 2, 2);

y_train = zeros(num_classes, n_train);
y_test = zeros(num_classes, n_test);

for i = 1:n_train
    y_train(train_labels(i), i) = 1; 
end

for i = 1:n_test
    y_test(test_labels(i), i) = 1; 
end

Krange = 1;
err = zeros (1, Krange);

for K = 1:Krange
    IDX = knnsearch(x_train, x_test,'Distance', 'minkowski', 'P', 1, 'K', K);
    class_votes = zeros(num_classes, n_test);
    predictions = zeros(num_classes, n_test);

    for i = 1:K
        class_votes = class_votes + y_train(:, IDX(:, i));
    end

    for i = 1:n_test
        predictions(:, i) = (class_votes(:, i) == max(class_votes(:, i)));
    end

    [err(K), ~, ~, ~] = confusion(y_test, predictions);
end

plot(err);

[minErr, bestK] = min(err);