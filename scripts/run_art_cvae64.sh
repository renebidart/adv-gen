python train_art_cvae.py --files_dict_loc /media/rene/data/art/files_dict.pkl --SAVE_PATH /media/rene/data/art/models/cvae --base_path /media/rene/data/art/train_64 --net_type CVAE_ART --IM_SIZE 64 --latent_size 128 --batch_size 150 --device "cuda:0" --lr .0005 --epochs 200

python train_art_cvae.py --files_dict_loc /media/rene/data/art/files_dict.pkl --SAVE_PATH /media/rene/data/art/models/cvae --base_path /media/rene/data/art/train_64 --net_type CVAE_ART --IM_SIZE 64 --latent_size 256 --batch_size 150 --device "cuda:0" --lr .0005 --epochs 200
