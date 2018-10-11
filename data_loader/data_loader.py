import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd

def load_dataset(path):
    data_path = path
    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return data_loader


# class mnistmTrainingDataset(torch.utils.data.Dataset):

#     def __init__(self,text_file,root_dir,transform=transformMnistm):
#         """
#         Args:
#             text_file(string): path to text file
#             root_dir(string): directory with all train images
#         """
#         self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
#         self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.name_frame)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
#         image = Image.open(img_name)
#         image = self.transform(image)
#         labels = self.label_frame.iloc[idx, 0]
#         #labels = labels.reshape(-1, 2)
#         sample = {'image': image, 'labels': labels}

#         return sample


# mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt',
#                                    root_dir = 'Downloads/mnist_m/mnist_m_train')

# mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)

# class footwearClassificationDataset(torch.utils.data.Dataset):

#     def __init__(self,text_file,root_dir,transform=transformMnistm):
#         """
#         Args:
#             text_file(string): path to text file
#             root_dir(string): directory with all train images
#         """
#         self.name_frame = pd.read_csv(text_file,sep=" ",usecols=range(1))
#         self.label_frame = pd.read_csv(text_file,sep=" ",usecols=range(1,2))
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.name_frame)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
#         image = Image.open(img_name)
#         image = self.transform(image)
#         labels = self.label_frame.iloc[idx, 0]
#         #labels = labels.reshape(-1, 2)
#         sample = {'image': image, 'labels': labels}

#         return sample


# mnistmTrainSet = mnistmTrainingDataset(text_file ='Downloads/mnist_m/mnist_m_train_labels.txt',
#                                    root_dir = 'Downloads/mnist_m/mnist_m_train')

# mnistmTrainLoader = torch.utils.data.DataLoader(mnistmTrainSet,batch_size=16,shuffle=True, num_workers=2)