caffe.reset_all();
caffe.set_mode_gpu();

batch_size = 64;
discriminator_iter = 5;
show_iter = 10;
image_height = 28;
image_width = 28;

generator_solver = caffe.Solver('models/generator_solver.prototxt');
discriminator_solver = caffe.Solver('models/discriminator_solver.prototxt');

all_err_d = [];

for iter = 1:10000
    for di = 1:discriminator_iter
        discriminator_solver.net.net_clear_param_diff();
        rand_vec = randn(100, batch_size);
        g = generator_solver.net.forward({rand_vec});
        d = discriminator_solver.net.forward(g);
        err_real_d = mean(d{1}(1:batch_size));
        err_gen_d = mean(d{1}(batch_size+1:2*batch_size));
        err_d = err_real_d - err_gen_d;
        d_back_diff = [-1/batch_size * ones(1, batch_size) 1/batch_size * ones(1, batch_size)];
%         d_back_diff = [log(d{1}(1:batch_size)) log(1-d{1}(batch_size+1:2*batch_size))];
        discriminator_solver.net.backward({d_back_diff});
        discriminator_solver.update();
    end;
    generator_solver.net.net_clear_param_diff();
    rand_vec = randn(100, batch_size);
    g = generator_solver.net.forward({rand_vec});
    d = discriminator_solver.net.forward(g);
    err_g = mean(d{1});
    d_back_diff = [zeros(1, batch_size) -1/batch_size * ones(1, batch_size)];
%     d_back_diff = [log(d{1}(1:batch_size)) log(1 - d{1}(batch_size+1:2*batch_size))];
    g_back_diff = discriminator_solver.net.backward({d_back_diff});
    generator_solver.net.backward(g_back_diff);
    generator_solver.update();
    fprintf('iter %d, err_d=%f, err_g=%f, err_real_d=%f, err_gen_d=%f\r\n', iter, err_d, err_g, err_real_d, err_gen_d);
    all_err_d = [all_err_d err_d];
    if mod(iter, show_iter) == 0
        figure(1);
        imshow(uint8(show_grid_image(g{1})));
        figure(2);
        plot(1:iter, all_err_d);
    end;
end;