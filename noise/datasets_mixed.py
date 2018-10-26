from salad.datasets.transforms.noise import Gaussian, SaltAndPepper
import torch
import salad.datasets
from torchvision import transforms, datasets

def snp2gauss(small = False):
    noisemodels = [
        SaltAndPepper(.36),
        Gaussian(0, .025),
        Gaussian(0, .05),
        Gaussian(0, .075),
        Gaussian(0, .1),
        Gaussian(0, .15),
        Gaussian(0, .2),
        Gaussian(0, .25),
        Gaussian(0, .3),
    ]
    if small:
        noisemodels = [
            noisemodels[0],
            noisemodels[5]

        ]
    return noisemodels

def gauss2snp(small = False):
    noisemodels = [
        Gaussian(0, .15),
        SaltAndPepper(.0025),
        SaltAndPepper(.01),
        SaltAndPepper(.05),
        SaltAndPepper(.1),
        SaltAndPepper(.15),
        SaltAndPepper(.2),
        SaltAndPepper(.25),
        SaltAndPepper(.36),
    ]
    if small:
        noisemodels = [
            noisemodels[0],
            noisemodels[-1]
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