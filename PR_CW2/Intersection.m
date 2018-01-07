function  [nn_idx] = Intersection(TrainData, TestData, K)        
    %find nearest neighbour
    Inter_distance = zeros(size(TestData,1),size(TrainData,1));
    nn_idx = zeros(size(TestData,1), K);
    for i=1:1:size(TestData,1)
       for j=1:1:size(TrainData,1) 
           % calculate the Chi2 distance between the test data and the train
           % data
           Inter_distance(i,j) = histogram_intersection_d_norm(TestData(i,:),TrainData(j,:));
       end
       
       for k = 1:K
           [~, matched_index] = min(Inter_distance(i,:));
           nn_idx(i, k) = matched_index;
           Inter_distance(i, matched_index) = intmax('int32');
       end
    end       
end