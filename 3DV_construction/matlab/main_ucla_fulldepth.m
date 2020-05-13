data_path = '/data/Datasets/NUCLA3D';
%bounding_box_path = '/data/wangyancheng/Vixel_Dynamic/Preprocess/all_bounding_box_ucla'
%output_path = '/data/wangyancheng/Vixel_Dynamic/pointcloudData/ntu/';
save_pc_dir = '/data/wangyancheng/Vixel_Dynamic/pointcloudData/UCLA_pc_m_rawdpeth_100_feature_2048/';
fx_d = 525.0
fy_d = 525.0
cx_d = 160.5
cy_d = 100.5
vox_size =80  % grid size(mm)
vox_num_x = 100; % grid size(mm)

SAMPLE_NUM = 2048;
sample_num_level1 = 512;
sample_num_level2 = 128;

v_list = dir([data_path,'/a*']);
for i = 1:length(v_list)
    VideoName = v_list(i).name
    png_th = dir([data_path,'/',VideoName,'/*.png']);
    
%     if exist([save_pc_dir VideoName '_xyzC.mat'],'file')
%         continue;
%     end
    
    n_frame = length(png_th);
    max_x = -5550;
    max_y = -5550;
    max_z = -5550;
    min_x = 5550;
    min_y = 5550;
    min_z = 55550;
    len_frame = zeros(n_frame,1);
    all_frames_points = [];
    
    %% load data and convert to points
    for i_frame = 1:n_frame
        depth_image = double(imread([data_path,'/',VideoName,'/',png_th(i_frame).name]));
        % convert depth to xyz
        [cloud, ordered]= depth2point(depth_image, fx_d, fy_d,cx_d,cy_d);

        figure(1),scatter3(cloud(:,1),cloud(:,2),cloud(:,3),'.','r');
        len_frame(i_frame) = size(cloud,1);
        all_frames_points = [all_frames_points;cloud];
        
        [p_max,index] = max(cloud);
        [p_min,indexd] = min(cloud);
        p_max_x = p_max(1);
        p_max_y = p_max(2);
        p_max_z = p_max(3);
        p_min_x = p_min(1);
        p_min_y = p_min(2);
        p_min_z = p_min(3);
        if(p_max_x>max_x)
            max_x = p_max_x;
        end
        if(p_max_y>max_y)
            max_y = p_max_y;
        end
        if(p_min_x<min_x)
            min_x = p_min_x;
        end
        if(p_min_y<min_y)
            min_y = p_min_y;
        end
        
        if(p_min_z<min_z)
            min_z = p_min_z;
        end
        if p_max_z>max_z
            max_z = p_max_z;
        end
    end
    % boundings of all points
    range_xyz = [max_x,max_y,max_z,min_x,min_y,min_z];
    
    %%  point  --->  voxel  --->  DI voxel
    %magifier = vox_num_x/(max_x - min_x);
    dx = ceil((max_x+0.1 - min_x)/vox_size)+1
    dy = ceil((max_y+0.1 - min_y)/vox_size)+1
    dz = ceil((max_z+0.1 - min_z)/vox_size)+1
    
    M = 5;% temporal split: 1(global)+4(temporal splits)
    voxel_DI = zeros(M, dx,dy,dz);
    start_idx = 1;
    
    for i_frame = 1:n_frame
        thereD_matrix = zeros(dx,dy,dz);
        point_f = all_frames_points(start_idx:start_idx+len_frame(i_frame)-1,:);
        % voxlization
        voxel_f = point2voxel_voxel(point_f, vox_size, thereD_matrix, range_xyz);
        % to ensure all voxel size are the same with dx,dy,dz
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
    
    % voxels to points to fit pointnet++
    tic
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
    toc

    
    
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
    %         if ~exist([save_pc_dir,VideoName],'dir')
    %            mkdir([save_pc_dir,VideoName])
    %         end
    if ~exist([save_pc_dir VideoName '_xyzC.mat'],'file')
        save([save_pc_dir VideoName '_xyzC.mat'],'pc')
    end
    
    
    
    
end

