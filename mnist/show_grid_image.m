function [ background ] = show_grid_image( blob_data )
%SHOW_GRID_IMAGE 此处显示有关此函数的摘要
%   此处显示详细说明
    sizeB = size(blob_data);
    GridL = ceil(sqrt(sizeB(4)));
    blob_data = blob_data * 128 + 128;
    scale = 1;
    border = 2;
    sizeB(1) = sizeB(1) * scale;
    sizeB(2) = sizeB(2) * scale;

    background = zeros((border+sizeB(1))*GridL+border,(border+sizeB(2))*GridL+border,sizeB(3));
    for i = 1:sizeB(4)
        x = ceil(i / GridL);
        y = mod(i - 1,GridL) + 1;
        patch = imresize(blob_data(:,:,:,i),[sizeB(1) sizeB(2)],'nearest');
        patch = permute(patch,[2 1 3]);
        background(border + (x-1)*(border+sizeB(1)) + 1 : x*(border+sizeB(1)),border + (y-1)*(border+sizeB(2)) + 1 : y*(border+sizeB(2)),:) = patch;
    end;
end

