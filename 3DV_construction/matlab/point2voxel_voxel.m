function thereD_matrix = point2voxel_voxel(points, voxel_size,thereD_matrix,range_xyz)
lx = range_xyz(4);
ly = range_xyz(5);
lz = range_xyz(6);
% hx = range_xyz(1);
% hy = range_xyz(2);
% hz = range_xyz(3);
%magifier = co/(hx-lx);
for itm = 1:size(points,1)
    x = points(itm,1);
    y = points(itm,2);
    z = points(itm,3);
    if x-lx<0 | y-ly<0 | z-lz<0
        continue;
    end
    thereD_matrix(ceil((x-lx)/voxel_size)+1,ceil((y-ly)/voxel_size)+1,ceil((z-lz)/voxel_size)+1) = 1;
    
end
%size(thereD_matrix)