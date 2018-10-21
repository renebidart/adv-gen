python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/CVAE-1_16_32_64-16-MNIST-10_model_best.pth.tar --model_type cvae --dataset MNIST --num_times 10 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/CVAE-1_16_32_64-16-MNIST-10_model_best.pth.tar --model_type cvae --dataset MNIST --num_times 25 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/CVAE-1_16_32_64-16-MNIST-10_model_best.pth.tar --model_type cvae --dataset MNIST --num_times 50 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/CVAE-1_16_32_64-16-MNIST-10_model_best.pth.tar --model_type cvae --dataset MNIST --num_times 100 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward

python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE-1_16_32_64-16-MNIST/VAE-1_16_32_64-16-MNIST --model_type vae --dataset MNIST --num_times 10 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE-1_16_32_64-16-MNIST/VAE-1_16_32_64-16-MNIST --model_type vae --dataset MNIST --num_times 25 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE-1_16_32_64-16-MNIST/VAE-1_16_32_64-16-MNIST --model_type vae --dataset MNIST --num_times 50 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE-1_16_32_64-16-MNIST/VAE-1_16_32_64-16-MNIST --model_type vae --dataset MNIST --num_times 100 --iterations 50 --latent_size 16 --df_sample_num 200 --device cuda:1 --deterministic_forward

python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE_ABS/VAE_ABS--8-MNIST --model_type vae --dataset MNIST --num_times 10 --iterations 50 --latent_size 8 --df_sample_num 200 --IM_SIZE 28 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE_ABS/VAE_ABS--8-MNIST --model_type vae --dataset MNIST --num_times 25 --iterations 50 --latent_size 8 --df_sample_num 200 --IM_SIZE 28 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE_ABS/VAE_ABS--8-MNIST --model_type vae --dataset MNIST --num_times 50 --iterations 50 --latent_size 8 --df_sample_num 200 --IM_SIZE 28 --device cuda:1 --deterministic_forward
python get_gen_perf.py --files_df_loc /media/rene/data/adv_gen/MNIST/mnist_normal/files_df.pkl --model_loc /media/rene/data/adv_gen/MNIST/mnist_normal/models/VAE_ABS/VAE_ABS--8-MNIST --model_type vae --dataset MNIST --num_times 100 --iterations 50 --latent_size 8 --df_sample_num 200 --IM_SIZE 28 --device cuda:1 --deterministic_forward