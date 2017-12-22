clear;
close all;

load face.mat;

width = 56;
height = 46;
num_classes = 52;
class_size = 10;

train_examples = 7;
test_examples = class_size - train_examples;
PCA_DIM = 150;

%Training features and labels
x_train = zeros(width*height, train_examples*num_classes);
x_test = zeros(width*height, test_examples*num_classes);

y_test = zeros(test_examples*num_classes, num_classes);

%Partitioning into training and testing data
for i = 1:num_classes
    y_test(1+(i-1)*test_examples : i*test_examples, i) = ones(test_examples, 1);
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

A = bsxfun(@minus, x_train, mean(x_train, 2));
x_train_norm = bsxfun(@rdivide, A, mean(std(x_train)));
B = bsxfun(@minus, x_test, mean(x_train, 2));
x_test_norm = bsxfun(@rdivide, B, mean(std(x_train)));

%Getting PCA features:
PCA = pca(x_train_norm, PCA_DIM);
PCA_train =  transpose(PCA.W)*x_train_norm;
PCA_test =  transpose(PCA.W)*x_test_norm;

%faceVec = PCA.W(:,1);
%range = max(faceVec) - min(faceVec)
%faceVec = faceVec - min(faceVec)
%faceVec = faceVec*(255.0/range)
%imshow(uint8(vec2mat(faceVec, width)));

%Training 1 vs the rest Multi-class SVM
y_train = -1 + (zeros(train_examples*num_classes, num_classes));
scores = zeros(num_classes*test_examples, num_classes);
scores_PCA = zeros(num_classes*test_examples, num_classes);

time_1vRest = zeros(num_classes, 1);
tic;

%# grid of parameters
folds = 5;
[C,gamma] = meshgrid(-2:1:2, -5:0.5:-3);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
y_train(1:train_examples, 1) = ones(train_examples, 1);

for i=1:numel(C)
    cv_acc(i) = svmtrain(y_train(:, 1), x_train_norm', sprintf('-c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
end

% for i = 1:num_classes
%     y_train(1+(i-1)*train_examples:i*train_examples, i) = ones(train_examples, 1);
%     crossval_err = svmtrain(y_train(:, i), x_train_norm', '-t 2 -g 10 -v 7');
%     crossval_err_PCA = svmtrain(y_train(:, i), transpose(PCA_train), '-t 2 -v 7');
%     lol = 0;
% end

for i = 1:num_classes
    y_train(1+(i-1)*train_examples:i*train_examples, i) = ones(train_examples, 1);
    model = svmtrain(y_train(:, i), x_train_norm', '-t 2');
    model_PCA = svmtrain(y_train(:, i), PCA_train', '-t 2');
    %Get all image scores from classifier i
    [prediction, accuracy, imgScores] = svmpredict(zeros(num_classes*test_examples, 1), x_test_norm', model, '-q');
    [prediction, accuracy, imgScores_PCA] = svmpredict(zeros(num_classes*test_examples, 1), PCA_test', model_PCA, '-q');
    %Normalize the scores to be in the range [0, 1]
    scores(:, i) = (imgScores - min(imgScores))/(max(imgScores)-min(imgScores));
    scores_PCA(:, i) = (imgScores_PCA - min(imgScores_PCA))/(max(imgScores_PCA)-min(imgScores_PCA));
    time_1vRest(i) = toc;
end

plot(time_1vRest, (1:size(time_1vRest, 1))/size(time_1vRest, 1));

%Testing 1 vs the rest Multi-class SVM
class_predictions = zeros(num_classes*test_examples, num_classes);
class_predictions_PCA = zeros(num_classes*test_examples, num_classes);

%Getting the highest output over all models for each image
for i = 1:num_classes*test_examples
    class_predictions(i, :) = (max(scores(i, :)) == scores(i, :));
    class_predictions_PCA(i, :) = (max(scores_PCA(i, :)) == scores_PCA(i, :));
end

%plotconfusion(y_test(1:30,1:10)',class_predictions(1:30,1:10)');
[error_1vRest,cm_1vRest,ind,per] = confusion(y_test',class_predictions');
[error_1vRest_PCA,cm_1vRest_PCA,ind,per] = confusion(y_test',class_predictions_PCA');

figure;
heatmap(cm_1vRest);

%1 vs one Multi-class SVM
class_votes = zeros(num_classes*test_examples, num_classes);
class_votes_PCA = zeros(num_classes*test_examples, num_classes);

time_1v1 = ones(num_classes*(num_classes-1)/2, 1);
index = 1;
tic
for i = 1:(num_classes-1)
    for j = (i+1):num_classes
        model = svmtrain([ones(train_examples, 1); (-1)*ones(train_examples, 1)], transpose([x_train(:, 1+(i-1)*train_examples:i*train_examples), x_train(:, 1+(j-1)*train_examples:j*train_examples)]), '-t 0');
        model_PCA = svmtrain([ones(train_examples, 1); (-1)*ones(train_examples, 1)], transpose([PCA_train(:, 1+(i-1)*train_examples:i*train_examples), PCA_train(:, 1+(j-1)*train_examples:j*train_examples)]), '-t 0');
        [prediction, accuracy, imgScores] = svmpredict(zeros(num_classes*test_examples, 1), transpose(x_test), model);
        [prediction_PCA, accuracy, imgScores] = svmpredict(zeros(num_classes*test_examples, 1), transpose(PCA_test), model_PCA);
        class_votes(:, i) = class_votes(:, i) + (1 + prediction)/2;
        class_votes(:, j) = class_votes(:, j) + (1 + (-1)*prediction)/2;
        class_votes_PCA(:, i) = class_votes(:, i) + (1 + prediction_PCA)/2;
        class_votes_PCA(:, j) = class_votes(:, j) + (1 + (-1)*prediction_PCA)/2;
        time_1v1(index) = toc;
        index = index + 1;
    end
end

figure;
plot(time_1v1, 100*(1:size(time_1v1, 1))/size(time_1v1, 1));
hold on;
plot(time_1vRest, 100*(1:size(time_1vRest, 1))/size(time_1vRest, 1));
legend('1 vs 1', '1 vs all');
xlabel('time [s]') % x-axis label
ylabel('% of training and testing finished') % y-axis label

for i = 1:num_classes*test_examples
    class_predictions(i, :) = max(class_votes(i, :)) == class_votes(i, :);
    class_predictions_PCA(i, :) = max(class_votes_PCA(i, :)) == class_votes_PCA(i, :);
end

[error_1v1,cm_1v1,ind,per] = confusion(y_test',class_predictions');
[error_1v1_PCA,cm_1v1_PCA,ind,per] = confusion(y_test',class_predictions_PCA');

figure;
heatmap(cm_1v1);
