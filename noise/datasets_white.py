from salad.datasets.transforms.noise import Gaussian
import torch
import salad.datasets
from torchvision import transforms, datasets

def clean2noise():
    noisemodels = [
        Gaussian(0, .001),
        Gaussian(0, .025),
        Gaussian(0, .05),
        Gaussian(0, .075),
        Gaussian(0, .1),
        Gaussian(0, .15),
        Gaussian(0, .2),
        Gaussian(0, .25),
        Gaussian(0, .3),
    ]
    return noisemodels

def noise2clean():
    noisemodels = [
        Gaussian(0, .3),
        Gaussian(0, .25),
        Gaussian(0, .2),
        Gaussian(0, .15),
        Gaussian(0, .1),
        Gaussian(0, .075),
        Gaussian(0, .05),
        Gaussian(0, .025),
        Gaussian(0, .001),
    ]
    return noisemodels


def get_dataset(noisemodels, batch_size, shuffle = True, num_workers = 0, which='train'):
    data = []
    
    for N in noisemodels:
    
        transform = transforms.Compose([
                transforms.ToTensor(),
                N,
                transforms.Normalize(mean=(0.43768448, 0.4437684,  0.4728041 ),
                                    std= (0.19803017, 0.20101567, 0.19703583))
        ])
        svhn = datasets.SVHN('/tmp/data', split=which, download=True, transform=transform)

        data.append(torch.utils.data.DataLoader(
            svhn, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers))

    loader = salad.datasets.JointLoader(*data)
    return data, loader