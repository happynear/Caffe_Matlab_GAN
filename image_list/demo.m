caffe.reset_all();
caffe.set_mode_gpu();
image_root= 'C:\datasets\CASIA-maxpy-clean-aligned-96';
image_list = 'C:\datasets\CASIA-maxpy-clean-aligned-96\list.txt';
% image_root= 'G:\CelebA-align';
% image_list = 'G:\CelebA-align\list_attr_celeba.txt';
image_list_pointer = 1;
image_list_fid = fopen(image_list,'r');
image_label = textscan(image_list_fid, '%s %d');
fclose(image_list_fid);
total_image = length(image_label{1});

batch_size = 64;
discriminator_iter = 1;
show_item = 1;
show_iter = 10;
image_height = 112;
image_width = 96;

generator_solver = caffe.Solver('models/generator_solver.prototxt');
discriminator_solver = caffe.Solver('models/discriminator_solver.prototxt');

for iter = 1:10000
    for di = 1:discriminator_iter
        discriminator_solver.net.net_clear_param_diff();
        rand_vec = randn(100, batch_size);
        image_data = zeros(image_width,image_height,3,batch_size, 'single');
        for b=1:batch_size
            image_filename = fullfile(image_root, image_label{1}{image_list_pointer});
            image_data(:,:,:,b) = caffe.io.load_image(image_filename);
            image_list_pointer = mod(image_list_pointer,total_image+1) + 1;
        end;
        image_data = (image_data - 128) / 128;
        g = generator_solver.net.forward({rand_vec});
        discriminator_input = zeros(image_width,image_height,3,batch_size * 2, 'single');
        discriminator_input(:,:,:,1:batch_size) = image_data;
        discriminator_input(:,:,:,batch_size+1:batch_size*2) = g{1};
        discriminator_solver.net.blobs('dis_input').reshape(size(discriminator_input));
        d = discriminator_solver.net.forward({discriminator_input});
        err_real_d = mean(d{1}(1:batch_size));
        err_gen_d = mean(d{1}(batch_size+1:2*batch_size));
        err_d = err_real_d - err_gen_d;
        d_back_diff = [-1/batch_size * ones(1, batch_size) 1/batch_size * ones(1, batch_size)];
        discriminator_solver.net.backward({d_back_diff});
        discriminator_solver.update();
    end;
    generator_solver.net.net_clear_param_diff();
    rand_vec = randn(100, batch_size);
    g = generator_solver.net.forward({rand_vec});
    discriminator_solver.net.blobs('dis_input').reshape(size(g{1}));
    d = discriminator_solver.net.forward(g);
    err_g = mean(d{1});
    d_back_diff = -1/batch_size * ones(1, batch_size);
    g_back_diff = discriminator_solver.net.backward({d_back_diff});
    generator_solver.net.backward(g_back_diff);
    generator_solver.update();
    fprintf('iter %d, err_d=%f, err_g=%f, err_real_d=%f, err_gen_d=%f\r\n', iter, err_d, err_g, err_real_d, err_gen_d);
    if iter == show_iter
        images = g{1};
        image_to_show = squeeze(g{1}(:,:,:,show_item));
        image_to_show = permute(image_to_show, [2 1 3]);
        image_to_show = image_to_show(:,:,[3 2 1]);
        image_to_show = image_to_show * 128 + 128;
        figure(1);
        imshow(uint8(image_to_show));
    end;
end;