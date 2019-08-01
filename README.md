# Pytorch implementation of Bidirectional One-Shot Unsupervised Domain Mapping.

## Run Training

### Download Dataset
To download dataset : bash datasets/download_cyclegan_dataset.sh $DATASET_NAME where DATASET_NAME is one of (monet2photo, summer2winter_yosemite)


### Phase I - train VAE for domain B

python train.py --dataroot=./datasets/summer2winter_yosemite/trainB --name=summer2winter_yosemite_autoencoder --model=autoencoder --dataset_mode=single


For reverse direction:

 python train.py --dataroot=./datasets/summer2winter_yosemite/trainA --name=summer2winter_yosemite_autoencoder_reverse --model=autoencoder --dataset_mode=single


### Phase II - Train domain A and domain B together

python train.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_biost --load_dir=summer2winter_yosemite_autoencoder --model=biost --start=0

For reverse direction:

python train.py --dataroot=./datasets/summer2winter_yosemite/ --name=summer2winter_yosemite_biost --load_dir=summer2winter_yosemite_autoencoder --model=biost --A='B' --B='A --start=0
