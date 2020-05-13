function sampled_idx = farthest_point_sampling_fast(point_cloud, sample_num)
% farthest point sampling
% point_cloud: Nx3
% author: Liuhao Ge

pc_num = size(point_cloud,1);

if pc_num <= sample_num
    sampled_idx = [1:pc_num]';
    sampled_idx = [sampled_idx; randi([1,pc_num],sample_num-pc_num,1)];
else
    sampled_idx = zeros(sample_num,1);
    sampled_idx(1) = randi([1,pc_num]);
    
    cur_sample = repmat(point_cloud(sampled_idx(1),:),pc_num,1);
    diff = point_cloud - cur_sample;
    min_dist = sum(diff.*diff, 2);

    for cur_sample_idx = 2:sample_num
        %% find the farthest point
        [~, sampled_idx(cur_sample_idx)] = max(min_dist);
        
        if cur_sample_idx < sample_num
            % update min_dist
            %valid_idx = find(min_dist>1e-8);
            %size(valid_idx)
            %diff = point_cloud(valid_idx,:) - repmat(point_cloud(sampled_idx(cur_sample_idx),:),sum(valid_idx),1);
            %min_dist(valid_idx,:) = min(min_dist(valid_idx,:), sum(diff.*diff, 2));
            diff = point_cloud - repmat(point_cloud(sampled_idx(cur_sample_idx),:),pc_num,1);
            min_dist = min(min_dist, sum(diff.*diff, 2));
        end
    end
end
sampled_idx = unique(sampled_idx);