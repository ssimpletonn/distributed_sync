import modin.pandas as pd
import ray
import pandas
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

def read_data_imagenet(train_path = './ILSVRC/Data/CLS-LOC/train', test_path = './content/imagenet-mini/val'):
    paths=[]
    labels=[]
    for dirname, _, filenames in os.walk('./ILSVRC/Data/CLS-LOC/train'):
        for filename in filenames:
            if filename[-4:]=='JPEG':
                paths+=[(os.path.join(dirname, filename))]
                label=dirname.split('/')[-1]
                labels+=[label]
    
    tpaths=[]
    tlabels=[]
    for dirname, _, filenames in os.walk('./content/imagenet-mini/val'):
        for filename in filenames:
            if filename[-4:]=='JPEG':
                tpaths+=[(os.path.join(dirname, filename))]
                label=dirname.split('/')[-1]
                tlabels+=[label]
    
    class_names=sorted(set(labels))
    N=list(range(len(class_names)))
    normal_mapping=dict(zip(class_names, N))
    reverse_mapping=dict(zip(N, class_names))

    df=pd.DataFrame(columns=['path','label'])
    df['path']=paths
    df['label']=labels
    df['label']=df['label'].map(normal_mapping)    
    
    tdf=pd.DataFrame(columns=['path','label'])
    tdf['path']=tpaths
    tdf['label']=tlabels
    tdf['label']=tdf['label'].map(normal_mapping)
    tdf.to_pickle("test_imagenet.pkl")

    df.to_pickle("train_dataset.pkl")
    tdf.to_pickle("test_dataset.pkl")

    return df, tdf


def create_dataloader_fromDF(df, tdf):
        
