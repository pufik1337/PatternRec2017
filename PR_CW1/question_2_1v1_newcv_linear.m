clear;
close all;

load face.mat;

width = 56;
height = 46;
num_classes = 52;
class_size = 10;

train_examples = 7;
test_examples = class_size - train_examples;
PCA_DIM = 100;
use_pca = 1;

%Training features and labels
x_train = zeros(width*height, train_examples*num_classes);
x_test = zeros(width*height, test_examples*num_classes);

y_test = zeros(test_examples*num_classes, num_classes);
y_train = -1 + (zeros(train_examples*num_classes, num_classes));

%Partitioning into training and testing data
for i = 1:num_classes
    y_test(1+(i-1)*test_examples : i*test_examples, i) = ones(test_examples, 1);
    y_train(1+(i-1)*train_examples:i*train_examples, i) = ones(train_examples, 1);
    %random shuffling
    X(:, 1 + (i-1)*class_size : i*class_size) = X(:, (i-1)*class_size + randperm(10));
    for j = 1:class_size
        if j <= train_examples
            x_train(:, (i-1)*train_examples + j) = X(:, (i-1)*class_size + j);
        else
            x_test(:, (i-1)*test_examples + (j-train_examples)) = X(:, (i-1)*class_size + j);
        end
    end       
end

k=7;                               %Define ratio of partition, k is the proportion sorted into test set

%Getting PCA features:
PCA = pca(x_train, PCA_DIM);
if use_pca == 1
    x_train =  transpose(PCA.W)*(x_train - PCA.mean_X);
    x_test =  transpose(PCA.W)*(x_test - PCA.mean_X);
    
end

C = (-5:0.5:5);
cv_acc = zeros(numel(C),1);

tic
parfor param=1:numel(C)
    err = zeros(k,1);
    for i = 1:k
        %Split data into the training and validation data
        TestIdx=mod(1-(i-1):size(x_train,2)-(i-1),7)<=1 & mod(1-(i-1):size(x_train,2)-(i-1),7)>0;                           
        TrainingIdx=not(TestIdx);       
        validation=x_train(:,TestIdx);              
        train=x_train(:,TrainingIdx);
        cv_examples = train_examples - 1;
        class_predictions = zeros(num_classes, num_classes);
        class_votes = zeros(num_classes, num_classes);
        index = 1;
        tic
        for n = 1:(num_classes-1)
            for j = (n+1):num_classes
                model = fitcsvm(([train(:, 1+(n-1)*cv_examples:n*cv_examples), ... 
                train(:, 1+(j-1)*cv_examples:j*cv_examples)])', ...
                [ones(cv_examples, 1); (-1)*ones(cv_examples, 1)], ...
                'Standardize',true,'KernelFunction','linear', 'KernelScale', 'auto', 'BoxConstraint', 10^C(param));
                [prediction, ~] = predict(model, validation');
                class_votes(:, n) = class_votes(:, n) + (1 + prediction)/2;
                class_votes(:, j) = class_votes(:, j) + (1 + (-1)*prediction)/2;
                index = index + 1;
            end
        end
        thistime = toc;
        thistime
        for m = 1:num_classes
            class_predictions(m, :) = (max(class_votes(m, :)) == class_votes(m, :));
        end
        [err(i),~,~,~] = confusion(0.5*(1+y_train(TestIdx, :))',class_predictions');
    end
    cv_acc(param) = sum(err)/size(err, 1);
    param
    cv_acc(param)
end    
cvtime_1v1 = toc;

[~,idx] = min(cv_acc);
best_C = 10^C(idx);

scores = zeros(num_classes*test_examples, num_classes);

%1 vs one Multi-class SVM
class_votes = zeros(num_classes*test_examples, num_classes);
avgSupportVectors = 0;
 
%index = 1;
tic;

for i = 1:(num_classes-1)
    for j = (i+1):num_classes
        model = fitcsvm(([x_train(:, 1+(i-1)*train_examples:i*train_examples), ... 
        x_train(:, 1+(j-1)*train_examples:j*train_examples)])', ...
        [ones(train_examples, 1); (-1)*ones(train_examples, 1)], ...
        'Standardize',true,'KernelFunction','linear', 'KernelScale',  'auto', 'BoxConstraint', 2.814);
        [prediction, ~] = predict(model, x_test');
        class_votes(:, i) = class_votes(:, i) + (1 + prediction)/2;
        class_votes(:, j) = class_votes(:, j) + (1 + (-1)*prediction)/2;
        %index = index + 1;
        avgSupportVectors = avgSupportVectors + (sum(model.IsSupportVector))/(0.5*num_classes*(num_classes-1));
    end
end

time_1v1 = toc;

avgSupportVectors = avgSupportVectors/(2*train_examples);
class_predictions = zeros(num_classes*test_examples, num_classes);

for i = 1:num_classes*test_examples
    class_predictions(i, :) = max(class_votes(i, :)) == class_votes(i, :);
end

[error_1v1,cm_1v1,~,~] = confusion(y_test',class_predictions');

imagesc(cm_1v1);
colorbar;

wrongIdx = 1;

for i = 1:num_classes*test_examples
    if ~isequal(y_test(i,:), class_predictions(i,:))
        wrongIdx = i;
        [~, predicted_class] = max(class_predictions(i, :));
        [~, true_class] = max(y_test(i, :));
    end
end
figure;
confused_imgs = [vec2mat(x_test(:,wrongIdx), width)', vec2mat(x_train(:,1 + (train_examples)*(true_class-1)), width)', vec2mat(x_train(:,1 + (train_examples)*(predicted_class-1)), width)'];
imshow(uint8(confused_imgs));

figure;
plot(C, cv_acc);
title('Cross-Validation error vs. log10(C); OvO SVM')
xlabel('log10(C)')
ylabel('Cross-Validation Error')
grid;
