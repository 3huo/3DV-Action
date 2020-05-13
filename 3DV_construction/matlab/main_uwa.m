data_path = '/UWA3D/UWA3DII_Depth/Actions_images';
bounding_box_path = '../UWA3D_person/result_txt';
save_pc_dir = 'UWA3D_point/UWA_vsize40_feature_2048/';
fx_d = 325.0
fy_d = 325.0
cx_d = 160.5
cy_d = 120.5
vox_size =40  % grid size(mm)
vox_num_x = 100; % grid size(mm)
if ~exist(save_pc_dir,'dir')
    mkdir(save_pc_dir);
end
SAMPLE_NUM = 2048;
sample_num_level1 = 512;
sample_num_level2 = 128;

v_list = dir([data_path,'/a*']);
for i = 1:length(v_list)
    VideoName = v_list(i).name
    %VideoName =  'a01_s03_e02_v02';
    png_th = dir([data_path,'/',VideoName,'/*.png']);
    n_frame = length(png_th);
    all_min_z = 55550;
    all_max_z = 55550;
    max_x = [];
    max_y = [];
    max_x_i = [];
    max_y_i = [];
    min_x_i = [];
    min_y_i = [];
    threshold = [];
    min_x = [];
    min_y = [];
    min_z = [];
    frame_use = [];
    len_frame = zeros(n_frame,1);
    all_frames_points = [];
%     
    p_max_zz = []; 
    
    %%
    for i_frame = 1:n_frame
        png_name = png_th(i_frame).name;
        depth_image = double(imread([data_path,'/',VideoName,'/',png_name]));%%depth image
        % load 2D bounding box
        bbox = [bounding_box_path,'/',VideoName,'/',png_name(1:end-4),'.txt']; %%2D bounding box for human
        try
            [object,p,y,x,w,h] = textread(bbox,'%s %s %s %s %s %s',1);
            x = x{1}; y = y{1}; w = w{1}; h = h{1};
            frame_use = [frame_use,i_frame];
        catch ME
            print = 'lose bbox!'
            continue
        end
        
        %% enlarge the size of detected bounding box
        x = str2double(x(1:5));
        y = str2double(y(2:6));
        w = str2double(w(1:5));
        h = str2double(h(1:5));
        x1 = x-h/2 -20; x2 = x+h/2 +20; y1 = y-w/2 -20; y2 = y+w/2 +10;
        if x1<1
            x1 = 1;end
        if x2>240
            x2 = 240;end
        if y1<1
            y1 = 1;end
        if y2>320
            y2 = 320;end
        x1 = round(x1);x2 = round(x2);y1 = round(y1);y2 = round(y2);
        
        % 2D mask for action
        exp = zeros(size(depth_image));
        exp(x1:x2,y1:y2) = 1;
        depth_image = exp.*depth_image;
        %convert image to points
        [cloud, ordered]= depth2point(depth_image, fx_d, fy_d,cx_d,cy_d);
        hh = cloud(:,3);%% z axis data
        
        % compute the threshold for human in each frame (To remove the background)
        [N,X] = hist(hh);
        num_hist = length(N);
        first_half = round(num_hist*0.7);
        [B, In] = sort(N(1:first_half),'descend');
        threshold = [threshold, X(In(1)+2)];
        
        % comput each human-point cloud bounding value 
        max_x = [max_x, max(cloud(:,1))];
        max_y = [max_y, max(cloud(:,2))];
        min_x = [min_x, min(cloud(:,1))];
        min_y = [min_y, min(cloud(:,2))];
        min_z = [min_z, min(cloud(:,3))];
        max_x_i = [max_x_i, x2];
        max_y_i = [max_y_i, y2];
        min_x_i = [min_x_i, x1];
        min_y_i = [min_y_i, y1];   
    end
    
    % compute the final 2D action bounding box with all frames
    min_x = min(min_x); min_y = min(min_y); min_z = min(min_z);max_x=max(max_z);max_y=max(max_y);
    min_x_i = min(min_x_i); min_y_i = min(min_y_i);max_x_i=max(max_x_i);max_y_i=max(max_y_i);
    % fix z axis threshold value for action
    threshold = sort(threshold,'descend');
    threshold = mean(threshold(1:round(0.3*length(threshold))));
    % final 3D bounding box
    range_xyz = [max_x,max_y,threshold,min_x,min_y,min_z];
    
    exp = zeros(size(depth_image));
    exp(min_x_i:max_x_i,min_y_i:max_y_i) = 1;%2d action mask
    % load depth to points
    for i_frame = 1:n_frame      
        depth_image = double(imread([data_path,'/',VideoName,'/',png_name]));
        depth_image = exp.*depth_image;
        [cloud, ordered]= depth2point(depth_image, fx_d, fy_d,cx_d,cy_d);
        %figure(2),imshow(depth_image)
        %figure(1),scatter3(cloud(:,1),cloud(:,2),cloud(:,3),'.','r');
        hh = cloud(:,3);
        cloud = cloud(hh<threshold,:);
        %figure(1),scatter3(cloud(:,1),cloud(:,2),cloud(:,3),'.','r');
        len_frame(i_frame) = size(cloud,1);
        all_frames_points = [all_frames_points;cloud];
    end
   
    %%  point  --->  voxel  --->  DI voxel
    %magifier = vox_num_x/(max_x - min_x);
    %voxel num x y z
    dx = ceil((max_x+0.1 - min_x)/vox_size)+1;
    dy = ceil((max_y+0.1 - min_y)/vox_size)+1;
    dz = ceil((max_z+0.1 - min_z)/vox_size)+1;
    
    M = 5;% temporal split
    voxel_DI = zeros(M, dx,dy,dz);
    start_idx = 1;
    
    for i_frame = 1:n_frame
        thereD_matrix = zeros(dx,dy,dz);
        point_f = all_frames_points(start_idx:start_idx+len_frame(i_frame)-1,:);
        
        voxel_f = point2voxel_voxel(point_f, vox_size, thereD_matrix, range_xyz);
        % voxlization
        if length(find(size(voxel_f)~=size(squeeze(voxel_DI(1,:,:,:)))))==1
            voxel_f = voxel_f(1:size(thereD_matrix,1),1:size(thereD_matrix,2),1:size(thereD_matrix,3));
        end
        % fast temporal rank pooling
        for m = 1:M
            if m==1
                voxel_DI(m,:,:,:) = squeeze(voxel_DI(m,:,:,:)) + (i_frame*2-n_frame-1)*voxel_f;
            end
            if (m ==2)&&(i_frame<=round(n_frame*2/5))
                voxel_DI(m,:,:,:) = squeeze(voxel_DI(m,:,:,:)) + (i_frame*2-round(n_frame*2/5)-1)*voxel_f;
            end
            if m==3&&(i_frame<=round(n_frame*3/5))&&(i_frame>round(n_frame*1/5))
                idx_f = i_frame-round(n_frame*1/5);
                len_f = round(n_frame*3/5) - round(n_frame*1/5);
                voxel_DI(m,:,:,:) = squeeze(voxel_DI(m,:,:,:)) + (idx_f*2-len_f-1)*voxel_f;
            end
            if m==4&&(i_frame<=round(n_frame*4/5))&&(i_frame>round(n_frame*2/5))
                idx_f = i_frame-round(n_frame*2/5);
                len_f = round(n_frame*4/5) - round(n_frame*2/5);
                voxel_DI(m,:,:,:) = squeeze(voxel_DI(m,:,:,:)) + (idx_f*2-len_f-1)*voxel_f;
            end
            if m==5&&(i_frame>round(n_frame*3/5))
                idx_f = i_frame-round(n_frame*3/5);
                len_f = n_frame - round(n_frame*2/5);
                voxel_DI(m,:,:,:) = squeeze(voxel_DI(m,:,:,:)) + (idx_f*2-len_f-1)*voxel_f;
            end
            
        end
        start_idx = start_idx+len_frame(i_frame);
    end
    %V_DI = reshape(voxel_DI,[6,])
    
    
    %tic
    % voxels to points to fit pointnet++
    xyz_c = [];
    for xx = 1:size(voxel_DI,2)
        for yy = 1:size(voxel_DI,3)
            for zz = 1:size(voxel_DI,4)
                if ~isempty(find(voxel_DI(:,xx,yy,zz)~=0, 1))
                    xyz_c = [xyz_c;[xx,yy,zz,squeeze(voxel_DI(:,xx,yy,zz)')]];
                    %feature = [feature;voxel_DI(xx,yy,zz)]
                end
            end
        end
    end
    %toc
    
   
    
    %% sample
     %sample(random will be much fast) fixed number points of each action
    if size(xyz_c,1)<SAMPLE_NUM
        tmp = floor(SAMPLE_NUM/size(xyz_c,1));
        rand_ind = [];
        for tmp_i = 1:tmp
            rand_ind = [rand_ind 1:size(xyz_c,1)];
        end
        rand_ind = [rand_ind randperm(size(xyz_c,1), mod(SAMPLE_NUM, size(xyz_c,1)))];
    else
        rand_ind = randperm(size(xyz_c,1),SAMPLE_NUM);
    end
    xyz_c_sample = xyz_c(rand_ind,:);
    
    %% Normalize Point Cloud
    x_min_max = [min(xyz_c(:,1)), max(xyz_c(:,1))];
    y_min_max = [min(xyz_c(:,2)), max(xyz_c(:,2))];
    z_min_max = [min(xyz_c(:,3)), max(xyz_c(:,3))];
    
    c_min_max = [min(xyz_c(:,4)), max(xyz_c(:,4))];
    c_len = c_min_max(2)-c_min_max(1);
    
    scale = 1.2;
    bb3d_x_len = scale*(x_min_max(2)-x_min_max(1));
    bb3d_y_len = scale*(y_min_max(2)-y_min_max(1));
    bb3d_z_len = scale*(z_min_max(2)-z_min_max(1));
    max_bb3d_len = bb3d_x_len;
    
    xyz_normalized_sample = xyz_c_sample(:,1:3)/max_bb3d_len;
    
    if size(xyz_c,1)<SAMPLE_NUM
        offset = mean(xyz_c(:,1:3))/max_bb3d_len;
    else
        offset = mean(xyz_normalized_sample);
    end
    
    xyz_normalized_sample = xyz_normalized_sample - repmat(offset,SAMPLE_NUM,1);
    xyz_c_normalized_sample = mapminmax(xyz_c_sample(:,4:8)')';
    %% FPS Sampling
    pc = [xyz_normalized_sample, xyz_c_normalized_sample];
    %pc = [hand_points_normalized_sampled normals_sampled_rotate];
    % 1st level
    sampled_idx_l1 = farthest_point_sampling_fast(pc(:,1:3), sample_num_level1)';
    other_idx = setdiff(1:SAMPLE_NUM, sampled_idx_l1);
    new_idx = [sampled_idx_l1 other_idx];
    pc = pc(new_idx,:);
    % 2nd level
    sampled_idx_l2 = farthest_point_sampling_fast(pc(1:sample_num_level1,1:3), sample_num_level2)';
    other_idx = setdiff(1:sample_num_level1, sampled_idx_l2);
    new_idx = [sampled_idx_l2 other_idx];
    pc(1:sample_num_level1,:) = pc(new_idx,:);
    %        if ~exist([output_path2,VideoName,'_xyz.txt'],'file')
    %            fid = fopen([output_path2,VideoName,'_xyz.txt'],'w');
    %            fprintf(fid,'%g\t',range_xyz);
    %            fclose(fid);
    %        end
    %        if ~exist([output_path2,VideoName,'_xyz.mat'],'file')
    %             save([output_path2,VideoName,'_xyz.mat'],'max_x','max_y','max_z','min_y','min_x','min_z')
    %        end
    %         %%  Save files
            if ~exist([save_pc_dir,VideoName],'dir')
               mkdir([save_pc_dir,VideoName])
            end
    [save_pc_dir VideoName '_xyzC.mat']
        save([save_pc_dir VideoName '_xyzC.mat'],'pc')
    
    
    
    
    
end

