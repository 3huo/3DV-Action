data_path = 'ntu120dataset';
%output_path = '/Vixel_Dynamic/pointcloudData/ntu/';
save_pc_dir = '/3DV_construction/pointcloudData/NTU_voxelz40_feature_2048_1motion/';
fx_d = 365.481
fy_d = 365.481
cx_d = 257.346
cy_d = 210.347
vox_size =40 % grid size(mm)
K = 80
feature_num = 4
SAMPLE_NUM = 2048;
sample_num_level1 = 512;
sample_num_level2 = 128;

if ~exist(save_pc_dir,'dir')
    mkdir(save_pc_dir)
end

for i = 1:32
    p_th = [data_path,'/nturgbd_depth_masked_s',num2str(i,'%03d'),'/nturgb+d_depth_masked']
    v_list = dir([p_th,'/S*'])
    
    for j = 1:length(v_list)
        
       
        
        VideoName = v_list(j).name
        png_th = dir([p_th,'/',VideoName,'/*.png']);
        
        if exist([save_pc_dir VideoName '_xyzC.mat'],'file')
            continue;
        end
        
        n_frame = length(png_th);
        max_x = -5550;
        max_y = -5550;
        max_z = -5550;
        min_x = 5550;
        min_y = 5550;
        min_z = 55550;
        len_frame = zeros(length(png_th),1);
        all_frames_points = [];
        
        if n_frame>=K
        frame_index = randperm(n_frame,K);
        n_frame = K;
        else
            frame_index = randperm(n_frame,n_frame);
            disp('error')
        end
        frame_index = sort(frame_index);
        
        
        
        tic
        for k = 1:n_frame
        
            png_th(frame_index(k)).name;
            depth_image = double(imread([p_th,'/',VideoName,'/',png_th(frame_index(k)).name]));
            %% convert depth to xyz
            [cloud, ordered]= depth2point(depth_image, fx_d, fy_d,cx_d,cy_d);
            %figure(1),scatter3(cloud(:,1),cloud(:,2),cloud(:,3),'.','r');
            %S002C001P010R002A060
            len_frame(k) = size(cloud,1);
            all_frames_points = [all_frames_points;cloud];
            
            %%
            
            %%
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
            if(p_max_z>max_z)
                max_z = p_max_z;
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
            
            %            if ~exist([output_path,VideoName],'dir')
            %                mkdir([output_path,VideoName])
            %            end
            %            if ~exist([output_path,VideoName,'/',png_th(k).name(8:end-4),'.txt'],'file')
            %                dlmwrite([output_path,VideoName,'/',png_th(k).name(8:end-4),'.txt'],cloud,'delimiter',' ');
            %            end
        end
        toc
        max_x = round(max_x);
        max_y = round(max_y);
        max_z = round(max_z);
%         min_y = floor(min_y);
%         min_x = floor(min_x);
%         min_z = floor(min_z);
        range_xyz = [max_x,max_y,max_z,min_x,min_y,min_z];
        
        %%  point  --->  voxel  --->  DI voxel
    dx = ceil((max_x+0.1 - min_x)/vox_size)+1;
    dy = ceil((max_y+0.1 - min_y)/vox_size)+1;
    dz = ceil((max_z+0.1 - min_z)/vox_size)+1;
        
        M = 1;
        voxel_DI = zeros(M, dx,dy,dz);
        start_idx = 1;
        tic
        for i_frame = 1:n_frame
            thereD_matrix = zeros(dx,dy,dz);
            point_f = all_frames_points(start_idx:start_idx+len_frame(i_frame)-1,:);
            
            voxel_f = point2voxel_voxel(point_f, vox_size, thereD_matrix, range_xyz);
            
            if length(find(size(voxel_f)~=size(squeeze(voxel_DI(1,:,:,:)))))==1
                voxel_f = voxel_f(1:size(thereD_matrix,1),1:size(thereD_matrix,2),1:size(thereD_matrix,3));
            end
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
                    len_f = n_frame - round(n_frame*3/5);
                    voxel_DI(m,:,:,:) = squeeze(voxel_DI(m,:,:,:)) + (idx_f*2-len_f-1)*voxel_f;
                end

            end
            start_idx = start_idx+len_frame(i_frame);
        end
        toc
        
        xyz_c = [];
        %feature =[]
        tic
        parfor xx = 1:size(voxel_DI,2)
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
        %%  compute surface normal
        
        %% Normalize Point Cloud
        x_min_max = [min(xyz_c(:,1)), max(xyz_c(:,1))];
        y_min_max = [min(xyz_c(:,2)), max(xyz_c(:,2))];
        z_min_max = [min(xyz_c(:,3)), max(xyz_c(:,3))];
           
        c_min_max = [min(xyz_c(:,4)), max(xyz_c(:,4))];
        c_len = c_min_max(2)-c_min_max(1);
        
        scale = 1.0;
        bb3d_x_len = scale*(x_min_max(2)-x_min_max(1));
        bb3d_y_len = scale*(y_min_max(2)-y_min_max(1));
        bb3d_z_len = scale*(z_min_max(2)-z_min_max(1));
        max_bb3d_len = bb3d_y_len;
        
        xyz_normalized_sample = xyz_c_sample(:,1:3)/max_bb3d_len;
        
        if size(xyz_c,1)<SAMPLE_NUM
            offset = mean(xyz_c(:,1:3))/max_bb3d_len;
        else
            offset = mean(xyz_normalized_sample);
        end
        
        xyz_normalized_sample = xyz_normalized_sample - repmat(offset,SAMPLE_NUM,1);
        xyz_c_normalized_sample = mapminmax(xyz_c_sample(:,4:feature_num)')';
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
    
end
