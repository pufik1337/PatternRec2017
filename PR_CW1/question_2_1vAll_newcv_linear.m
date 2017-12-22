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
use_pca = 0;

%Training features and labels
x_train = zeros(width*height, train_examples*num_classes);
x_test = zeros(width*height, test_examples*num_classes);

y_test = zeros(test_examples*num_classes, num_classes);
y_train = -1 + (zeros(train_examples*num_classes, num_classes));

%Partitioning into training and testing data
for i = 1:num_classes
    y_test(1+(i-1)*test_examples : i*test_examples, i) = ones(test_examples, 1);
    y_train(1+(i-1)*train_examples:i*train_examples, i) = ones(train_examples, 1);
    %Random shuffling to ensure the training/testing split is random
    X(:, 1 + (i-1)*class_size : i*class_size) = X(:, (i-1)*class_size + randperm(10));
    for j = 1:class_size
        if j <= train_examples
            x_train(:, (i-1)*train_examples + j) = X(:, (i-1)*class_size + j);
        else
            x_test(:, (i-1)*test_examples + (j-train_examples)) = X(:, (i-1)*class_size + j);
        end
    end       
end

%Getting PCA features:
PCA = pca(x_train, PCA_DIM);
if use_pca == 1
    x_train =  transpose(PCA.W)*(x_train - PCA.mean_X);
    x_test =  transpose(PCA.W)*(x_test - PCA.mean_X);
end

k=7;                               %Define ratio of partition, k is the proportion sorted into test set

C = (-4:0.1:3);
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

        scores = zeros(num_classes, num_classes);
        class_predictions = zeros(num_classes, num_classes);

        for j = 1:num_classes
            train_labels = y_train(TrainingIdx, j);
            model = fitcsvm(train', train_labels,'Standardize',true,'KernelFunction','linear','KernelScale', 'auto', 'BoxConstraint', 10^C(param));
            [~, imgScores] = predict(model, validation');
            scores(:, j) = (imgScores(:, 2) - min(imgScores(:, 2)))/(max(imgScores(:, 2))-min(imgScores(:, 2)));
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
best_C = 10^C(idx);

scores = zeros(num_classes*test_examples, num_classes);
scores_PCA = zeros(num_classes*test_examples, num_classes);

tic;

avgSupportVectors = 0;

randClass = randi(num_classes);
boundaries = zeros(width*height, num_classes);

%Training 1 vs the rest Multi-class SVM
for i = 1:num_classes
    model = fitcsvm(x_train', y_train(:, i),'Standardize',true,'KernelFunction','linear','KernelScale', 'auto', 'BoxConstraint', best_C);%100.52, 'BoxConstraint', 5.2289);
    %Get all image scores from classifier i
    [~, imgScores] = predict(model, x_test');
    %Normalize the scores to be in the range [0, 1]
    scores(:, i) = (imgScores(:, 2) - min(imgScores(:, 2)))/(max(imgScores(:, 2))-min(imgScores(:, 2)));
    avgSupportVectors = avgSupportVectors + (sum(model.IsSupportVector))/(num_classes);
    if i == randClass
        suppVecs = model.IsSupportVector;
    end
    boundaries(:, i) = (model.Alpha')*model.SupportVectors;    
end


time_1vRest = toc;

%Calculating the average number of support vectors per model
avgSupportVectors = avgSupportVectors/(num_classes*train_examples);

%Testing 1 vs the rest Multi-class SVM
class_predictions = zeros(num_classes*test_examples, num_classes);

%Getting the highest output over all models for each image
for i = 1:num_classes*test_examples
    class_predictions(i, :) = (max(scores(i, :)) == scores(i, :));
end

%Getting the test error and confusion matrix
[error_1vRest,cm_1vRest,~,~] = confusion(y_test',class_predictions');

%Plotting the confusion matrix
imagesc(cm_1vRest);
colorbar;

%Extracting a failure and success case
wrongIdx = 1;
goodIdx = 1;
for i = 1:num_classes*test_examples
    if ~isequal(y_test(i,:), class_predictions(i,:))
        wrongIdx = i;
        [~, predicted_class_fail] = max(class_predictions(i, :));
        [~, true_class_fail] = max(y_test(i, :));
    end
    if isequal(y_test(i,:), class_predictions(i,:))
        if i == randClass 
            goodIdx = i;
            [~, predicted_class_success] = max(class_predictions(i, :));
        end
    end    
end

%Visualizing the support vectors
SV_imgs_own =  x_train(:, suppVecs == y_train(:, randClass));
SV_imgs_rest = x_train(:, suppVecs == -(y_train(:, randClass)));

n_imgs = 4;
boundary_imgs = 8;

SV_image = zeros(width, height*n_imgs*2);
boundary_image = zeros(width, height*boundary_imgs);

for i = 1:min(n_imgs, (size(SV_imgs_own, 2)))
    SV_image(:, 1 + (i-1)*height:i*height) = vec2mat(SV_imgs_own(:, i), width)';
end

for i = 1:min(n_imgs, (size(SV_imgs_rest, 2)))
    j = i + n_imgs;
    SV_image(:, 1 + (j-1)*height:j*height) = vec2mat(SV_imgs_rest(:, randi(size(SV_imgs_rest, 2))), width)';
end
imshow(uint8(SV_image));
title(sprintf('Support vectors of class %d (OvA SVM)', randClass))

%Visualizing the boundary for the random class
bd_indices = randi(num_classes, boundary_imgs);
bd_imgs = boundaries(:, bd_indices);
for i = 1:boundary_imgs
    this_img = (bd_imgs(:, i) - min(bd_imgs(:, i)))*(255.0/(max(bd_imgs(:, i)) - min(bd_imgs(:, i))));
    boundary_image(:, 1 + (i-1)*height:i*height) = vec2mat(this_img, width)';
end
figure;
imshow(uint8(boundary_image));
title(sprintf('Decision boundaries of %d random classes (OvA SVM)', boundary_imgs));

%Showing the failure case images
figure;
confused_imgs = [vec2mat(x_test(:,wrongIdx), width)', vec2mat(x_train(:,1 + (train_examples)*(true_class_fail-1)), width)', vec2mat(x_train(:,1 + (train_examples)*(predicted_class_fail-1)), width)'];
imshow(uint8(confused_imgs));
title(sprintf('Classification Failure Example: True Class %d; Predicted Class %d', true_class_fail, predicted_class_fail));


%Showing the failure case images
figure;
correct_imgs = [vec2mat(x_test(:,goodIdx), width)', vec2mat(x_train(:,1 + (train_examples)*(predicted_class_success-1)), width)', vec2mat(x_train(:,2 + (train_examples)*(predicted_class_success-1)), width)'];
imshow(uint8(correct_imgs));
title(sprintf('Classification Success Example: Class %d', predicted_class_success));

%Plotting the cross-validation error as a function of log(C)
figure;
plot(C, cv_acc);
title('Cross-Validation Error vs. log10(C)')
xlabel('log10(c)')
ylabel('Cross-Validation Error')
grid;
