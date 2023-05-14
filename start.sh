# python3 train.py --task music --train_data /root/CoNAL-pytorch/data/Music/train --valid_data /root/CoNAL-pytorch/data/Music/valid --device cpu --input_dim 124 --n_class 11 --n_annotator 44
python3 train.py --task labelme --train_data /root/CoNAL-pytorch/data/LabelMe/train --valid_data /root/CoNAL-pytorch/data/LabelMe/valid --device cpu --input_dim 150528 --n_class 8 --n_annotator 59

python train.py --task music --train_data ./data/Music/train --valid_data ./data/Music/valid --device cpu --input_dim 124 --n_class 11 --n_annotator 44 --epochs 800 --batch_size 128
python train.py --task labelme --train_data ./data/LabelMe/train --valid_data ./data/LabelMe/valid --device cpu --input_dim 150528 --n_class 8 --n_annotator 59


python test.py  --device cpu  --batch_size 128 --test_data ./data/Music/test --device cpu --ckpt_dir ./checkpoints/
