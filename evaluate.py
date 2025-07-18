import pyiqa
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Evaluate Image Quality')

parser.add_argument('--input_dir', default='./results', type=str, help='Directory of validation images')

args = parser.parse_args()

brisque = pyiqa.create_metric('brisque')
nima = pyiqa.create_metric('nima', base_model_name='vgg16', train_dataset='ava', num_classes=10)
musiq = pyiqa.create_metric('musiq')
paq2piq = pyiqa.create_metric('paq2piq')


# Load images
dir_0 = args.input_dir

files = [x for x in os.listdir(dir_0) if x.endswith('.png')]
#
sum_nima = 0
sum_brisque = 0
sum_musiq = 0
sum_paq2piq = 0
count = 0

for file in tqdm(files):
    if(os.path.exists(os.path.join(dir_0,file))):
        # Load images
        if file.endswith('Store') or file.endswith('.txt'):
            continue
        image = os.path.join(dir_0, file)
        
        dist_brisque = brisque(image)
        dist_nima = nima(image)
        dist_musiq = musiq(image)
        dist_paq2piq = paq2piq(image)
        sum_brisque += dist_brisque
        sum_nima += dist_nima
        sum_musiq += dist_musiq
        sum_paq2piq += dist_paq2piq
        count += 1


print(dir_0)
print('Average BRISQUE: %.4f'%(sum_brisque/count))
print('Average NIMA: %.4f'%(sum_nima/count))
print('Average MUSIQ: %.4f'%(sum_musiq/count))
print('Average PAQ2PIQ: %.4f'%(sum_paq2piq/count))

       