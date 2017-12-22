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
y_train = -1 + (zeros(train_examples*num_classes, num_classes));

%Partitioning into training and testing data
for i = 1:num_classes
    y_test(1+(i-1)*test_examples : i*test_examples, i) = ones(test_examples, 1);
    y_train(1+(i-1)*train_examples:i*train_examples, i) = ones(train_examples, 1);
    %random shuffling
    %X(:, 1 + (i-1)*class_size : i*class_size) = X(:, (i-1)*class_size + randperm(10));
    for j = 1:class_size
        if j <= train_examples
            x_train(:, (i-1)*train_examples + j) = X(:, (i-1)*class_size + j);
        else
            x_test(:, (i-1)*test_examples + (j-train_examples)) = X(:, (i-1)*class_size + j);
        end
    end       
end

k=7;                               %Define ratio of partition, k is the proportion sorted into test set
c = cvpartition(num_classes*train_examples,'Kfold',k);       %Create partition object

[C,gamma] = meshgrid(-2:1:3, -5:2:5);
cv_acc = zeros(numel(C),1);

tic
for param=1:numel(C)
    err = zeros(k,1);
    parfor i = 1:k
        %Split data into the training and validation data
        TestIdx=test(c,i);                    
        TrainingIdx=training(c,i);            
        validation=x_train(:,TestIdx);              
        train=x_train(:,TrainingIdx);

        scores = zeros(num_classes, num_classes);
        class_predictions = zeros(num_classes, num_classes);

        for k = 1:(num_classes-1)
            for j = (i+1):num_classes
                model = fitcsvm(([x_train(:, 1+(k-1)*train_examples:k*train_examples), ... 
                x_train(:, 1+(j-1)*train_examples:j*train_examples)])', ...
                [ones(train_examples, 1); (-1)*ones(train_examples, 1)], ...
                'Standardize',true,'KernelFunction','linear', 'KernelScale', 10.934 , 'BoxConstraint', 0.25096);
        %         model = fitcsvm(([x_train(:, 1+(i-1)*train_examples:i*train_examples), ... 
        %         x_train(:, 1+(j-1)*train_examples:j*train_examples)])', ...
        %         [ones(train_examples, 1); (-1)*ones(train_examples, 1)],'OptimizeHyperparameters','auto',...
        %         'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        %         'expected-improvement-plus'));
                %model_PCA = svmtrain([ones(train_examples, 1); (-1)*ones(train_examples, 1)], transpose([PCA_train(:, 1+(i-1)*train_examples:i*train_examples), PCA_train(:, 1+(j-1)*train_examples:j*train_examples)]), '-t 0');
                [prediction, ~] = predict(model, x_test');
               % [prediction_PCA, accuracy, imgScores] = svmpredict(zeros(num_classes*test_examples, 1), transpose(PCA_test), model_PCA);
                class_votes(:, k) = class_votes(:, k) + (1 + prediction)/2;
                class_votes(:, j) = class_votes(:, j) + (1 + (-1)*prediction)/2;
                %class_votes_PCA(:, i) = class_votes(:, i) + (1 + prediction_PCA)/2;
                %class_votes_PCA(:, j) = class_votes(:, j) + (1 + (-1)*prediction_PCA)/2;
                index = index + 1;
            end
        end
        
        for m = 1:num_classes
            class_predictions(m, :) = (max(scores(m, :)) == scores(m, :));
        end
        [err(i),~,~,~] = confusion(0.5*(1+y_train(TestIdx, :))',class_predictions');
    end
    cv_acc(param) = sum(err)/size(err, 1);
    param
    cv_acc(param)
end    
cvtime = toc;

[~,idx] = min(cv_acc);
best_C = 5^C(idx);
best_gamma = 10^gamma(idx);

%Getting PCA features:
PCA = pca(x_train, PCA_DIM);
PCA_train =  transpose(PCA.W)*x_train;
PCA_test =  transpose(PCA.W)*x_test;

%faceVec = PCA.W(:,1);
%range = max(faceVec) - min(faceVec)
%faceVec = faceVec - min(faceVec)
%faceVec = faceVec*(255.0/range)
%imshow(uint8(vec2mat(faceVec, width)));

scores = zeros(num_classes*test_examples, num_classes);
scores_PCA = zeros(num_classes*test_examples, num_classes);

tic;

%Training 1 vs the rest Multi-class SVM
parfor i = 1:num_classes
    model = fitcsvm(x_train', y_train(:, i),'Standardize',true,'KernelFunction','RBF','KernelScale', best_gamma, 'BoxConstraint', best_C);%100.52, 'BoxConstraint', 5.2289);
    %model = fitcsvm(x_train',y_train(:, i),'OptimizeHyperparameters','auto',...
    %'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    %'expected-improvement-plus'));
    %model_PCA = fitcsvm(PCA_train', y_train(:, i));
    %Get all image scores from classifier i
    [~, imgScores] = predict(model, x_test');
    %[~, imgScores_PCA] = predict(model_PCA, PCA_test');
    %Normalize the scores to be in the range [0, 1]
    scores(:, i) = (imgScores(:, 2) - min(imgScores(:, 2)))/(max(imgScores(:, 2))-min(imgScores(:, 2)));
    %scores_PCA(:, i) = (imgScores_PCA(:, 2) - min(imgScores_PCA(:, 2)))/(max(imgScores_PCA(:, 2))-min(imgScores_PCA(:, 2)));
    i
end

time_1vRest = toc;

plot(time_1vRest, (1:size(time_1vRest, 1))/size(time_1vRest, 1));

%Testing 1 vs the rest Multi-class SVM
class_predictions = zeros(num_classes*test_examples, num_classes);
%class_predictions_PCA = zeros(num_classes*test_examples, num_classes);

%Getting the highest output over all models for each image
for i = 1:num_classes*test_examples
    class_predictions(i, :) = (max(scores(i, :)) == scores(i, :));
    %class_predictions_PCA(i, :) = (max(scores_PCA(i, :)) == scores_PCA(i, :));
end

[error_1vRest,cm_1vRest,ind,per] = confusion(y_test',class_predictions');
%[error_1vRest_PCA,cm_1vRest_PCA,ind,per] = confusion(y_test',class_predictions_PCA');

figure;
heatmap(cm_1vRest);

