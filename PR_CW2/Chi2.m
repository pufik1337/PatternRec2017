function  [nn_idx] = Chi2(TrainData, TestData, K)        
    %find nearest neighbour
    Chi2_distance = zeros(size(TestData,1),size(TrainData,1));
    nn_idx = zeros(size(TestData,1), K);
    for i=1:1:size(TestData,1)
       for j=1:1:size(TrainData,1) 
           % calculate the Chi2 distance between the test data and the train
           % data
           for dim=1:size(TestData,2)
               Chi2_distance(i,j) = 0.5 * (TestData(i,dim)-TrainData(j,dim))*(TestData(i,dim)-TrainData(j,dim))/(TestData(i,dim)+TrainData(j,dim)) + Chi2_distance(i,j);
           end
       end
       
       for k = 1:K
           [~, matched_index] = min(Chi2_distance(i,:));
           nn_idx(i, k) = matched_index;
           Chi2_distance(i, matched_index) = intmax('int32');
       end
    end       
end