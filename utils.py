import os
import pickle
import numpy as np
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import torch
import torchnet as tnt

#modify the data root
#_MINI_IMAGENET_DATASET_DIR = 'MiniImagenet'

def load_data(file):
    with open(file,'rb') as f:
        data=pickle.load(f,encoding='iso-8859-1')
    return data

def buildLabelIndex(labels):
    label2inds={}
    for idx,label in enumerate(labels):
        if label not in label2inds:
            label2inds[label]=[]
        label2inds[label].append(idx)
    return label2inds
    

class Radar_dataset(data.Dataset):
    def __init__(self,phase='train',do_not_use_random_transf=False):
        self.base_folder='miniImagenet'
        assert(phase=='train' or phase=='val' or phase=='test')
        self.phase=phase
        self.name='MiniImageNet_'+phase
        
        print('Loading mini ImageNet dataser - phase {0}'.format(phase))
        file_train_categories_train_phase = 'radar_pickle/train_train_normalize.pickle'
        file_train_categories_val_phase = 'radar_pickle/train_test_normalize.pickle'
        file_train_categories_test_phase = 'radar_pickle/train_test_normalize.pickle'
        file_val_categories_val_phase = 'radar_pickle/val_inferon_normalize.pickle'
        file_test_categories_test_phase = 'radar_pickle/val_inferon_normalize.pickle'

        
        if self.phase=='train':
            #during training phase we only load the training phase images of the training category
            data_train=load_data(file_train_categories_train_phase)
            self.data=data_train['data'] #array (n,84,84,3)
            self.labels=data_train['labels'] #list[n]
            
            self.label2ind=buildLabelIndex(self.labels)
            self.labelIds=sorted(self.label2ind.keys())
            self.num_cats=len(self.labelIds)
            self.labelIds_base=self.labelIds
            self.num_cats_base=len(self.labelIds_base)
        elif self.phase=='val' or self.phase=='test':
            if self.phase=='test':
                data_base=load_data(file_train_categories_test_phase)
                data_novel=load_data(file_test_categories_test_phase)
            else:
                data_base=load_data(file_train_categories_val_phase)
                data_novel=load_data(file_val_categories_val_phase)
            
            self.data=np.concatenate(
                [data_base['data'],data_novel['data']],axis=0)
            self.labels=list(data_base['labels'])+list(data_novel['labels'])
            
            self.label2ind=buildLabelIndex(self.labels)
            self.labelIds=sorted(self.label2ind.keys())
            self.num_cats=len(self.labelIds)
            
            self.labelIds_base=buildLabelIndex(data_base['labels']).keys()
            print('lb', self.labelIds_base)
            self.labelIds_novel=buildLabelIndex(data_novel['labels']).keys()
            print('ln', self.labelIds_novel)
            self.num_cats_base=len(self.labelIds_base)
            print('ncb', self.num_cats_base)
            self.num_cats_novel=len(self.labelIds_novel)
            print('ncn', self.num_cats_novel)
            intersection=set(self.labelIds_base) & set(self.labelIds_novel)
            assert(len(intersection)==0)
        else:
            raise ValueError('Not valid phase {0}'.fotmat(self.phase))

        if (self.phase=='test' or self.phase=='val') or (do_not_use_random_transf==True):
            self.transform=transforms.Compose([
                lambda x:np.array(x),
                transforms.ToTensor(),
                #normalize
            ])
        else:
            self.transform=transforms.Compose([
                transforms.RandomCrop(84,padding=8),
                #transforms.RandomHorizontalFlip(),
                lambda x : np.array(x),
                transforms.ToTensor(),
                #normalize
            ])
            
    def __getitem__(self,index):
        img,label=self.data[index],self.labels[index]
        #doing this so that it is consistent with all other datasets to return a PIL image
        img = img.squeeze()
        img=Image.fromarray(img)
        if self.transform is not None:
            img=self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.data)


class FewShotDataloader():
    def __init__(self,dataset,
                nKnovel=2,#number of novel categories
                nKbase=-1,#number of base categories
                nExemplars=1,#number of training examples per novel category
                nTestNovel=15*5,#number of test examples for all novel categories
                nTestBase=15*5,#number of test examples for all base categories
                batch_size=1,#number of training episodes per batch
                num_workers=4,
                epoch_size=2000):
        self.dataset=dataset
        self.phase=self.dataset.phase
        max_possible_nKnovel=(self.dataset.num_cats_base if self.phase=='train'
                             else self.dataset.num_cats_novel)
        print('kn', max_possible_nKnovel)
        #assert(nKnovel>=0 and nKnovel<max_possible_nKnovel)
        self.nKnovel=nKnovel
        
        max_possible_nKbase=self.dataset.num_cats_base
        nKbase=nKbase if nKbase>=0 else max_possible_nKbase


        print('kb', max_possible_nKbase)
        
        if self.phase=='train' and nKbase>0:
            nKbase-=self.nKnovel
            max_possible_nKbase-=self.nKnovel
            
        assert(nKbase>=0 and nKbase <=max_possible_nKbase)
        self.nKbase=nKbase
        
        self.nExemplars=nExemplars
        self.nTestNovel=nTestNovel
        self.nTestBase=nTestBase
        self.batch_size=batch_size
        self.epoch_size=epoch_size
        self.num_workers=num_workers
        self.is_eval_mode=(self.phase=='test') or (self.phase=='val')
        
    def sampleImageIdsFrom(self, cat_id , sample_size=1):
        """
        samples 'sample_size' number of unique image ids picked from the
        category 'cat_id'
        """
        assert(cat_id in self.dataset.label2ind)
        assert(len(self.dataset.label2ind[cat_id])>=sample_size)
        #Note : random.sample samples elements without replacement.
        return random.sample(self.dataset.label2ind[cat_id],sample_size)
    
    def sampleCategories(self,cat_set,sample_size=1):
        """
        Samples 'sample_size' number of unique categories picked from the
        'cat_set' set of categories.'cat_set' can be either 'base' or 'novel'.
        """
        if cat_set=='base':
            labelIds=self.dataset.labelIds_base
        elif cat_set=='novel':
            labelIds=self.dataset.labelIds_novel
        else:
            raise ValueError('Not recognize category set {}'.format(cat_set))
            
        assert(len(labelIds)>=sample_size)
        
        return random.sample(labelIds,sample_size)
    
    def sample_base_and_novel_categories(self,nKbase , nKnovel):
        """
        Samples 'nKbase' number of base categories and 'nKnovel'  number of novel categories.
        """
        if self.is_eval_mode:
            assert(nKnovel<=self.dataset.num_cats_novel)
            
            Kbase = sorted(self.sampleCategories('base',nKbase))
            Knovel = sorted(self.sampleCategories('novel',nKnovel))
        else:
            cats_ids = self.sampleCategories('base',nKbase+nKnovel)
            assert(len(cats_ids)==(nKbase+nKnovel))
            
            random.shuffle(cats_ids)
            Knovel=sorted(cats_ids[:nKnovel])
            Kbase=sorted(cats_ids[nKnovel:])
        return Kbase,Knovel
    
    def sample_test_examples_for_base_categories(self,Kbase,nTestBase):
        """
        Sample 'nTestBase' number of images from the 'Kbase' categories.
    
        """
        Tbase=[]
        if len(Kbase)>0:
            KbaseIndices=np.random.choice(np.arange(len(Kbase)),size=nTestBase,replace=True)
            KbaseIndices,NumImagesPerCategory=np.unique(KbaseIndices,return_counts=True)
            
            for Kbase_idx,NumImages in zip(KbaseIndices,NumImagesPerCategory):
                imd_ids=self.sampleImageIdsFrom(Kbase[Kbase_idx],sample_size=NumImages)
                Tbase+=[(img_id,Kbase_idx) for img_id in imd_ids]
        assert(len(Tbase)==nTestBase)
        
        return Tbase
    
    def sample_train_and_test_examples_for_novel_categories(
        self,Knovel,nTestNovel,nExemplars,nKbase):
        """
        Samples train and test examples of the novel categories.
        
        Args:
            Knovel:a list with the ids of the novel categories
            nTestNovel:the total number of test imgs that will be sampled from all novel categories
            nExemplars:the number of training examples per novel category that will be sampled
            nKbase:the number of base categories.it's used as offset of the category index of each sampled img
            
        Returns:
            Tnovel:a list of length 'nTestNovel' with 2-element tuple.
                    (img_id , category_label)
            Exemplars: a list of length len(Knovel)*nExemplars of 2-element tuple
                    (img_id , category_label range in [nKbase,nKbase+len(Knovel)-1])
        """
        if len(Knovel)==0:
            return [],[]
        
        nKnovel=len(Knovel)
        Tnovel=[]
        Exemplars=[]
        assert((nTestNovel % nKnovel)==0)
        nEvalExamplesPerClass = int(nTestNovel/nKnovel)
        
        for Knovel_idx in range(len(Knovel)):
            imd_ids=self.sampleImageIdsFrom(Knovel[Knovel_idx],sample_size=(nEvalExamplesPerClass+nExemplars))
            imds_tnovel=imd_ids[:nEvalExamplesPerClass]
            imds_ememplars=imd_ids[nEvalExamplesPerClass:]
            
            Tnovel+=[(img_id , nKbase+Knovel_idx) for img_id in imds_tnovel]
            Exemplars+=[(img_id,nKbase+Knovel_idx) for img_id in imds_ememplars]
        
        assert(len(Tnovel)==nTestNovel)
        assert(len(Exemplars)==len(Knovel)*nExemplars)
        random.shuffle(Exemplars)
        
        return Tnovel,Exemplars
    
    def sample_episode(self):
        """
        Sample a training episode
        """
        nKnovel=self.nKnovel
        nKbase=self.nKbase
        nTestNovel=self.nTestNovel
        nTestBase=self.nTestBase
        nExemplars=self.nExemplars
        
        Kbase,Knovel = self.sample_base_and_novel_categories(nKbase,nKnovel)
        Tbase=self.sample_test_examples_for_base_categories(Kbase,nTestBase)
        Tnovel,Exemplars=self.sample_train_and_test_examples_for_novel_categories(Knovel,nTestNovel,nExemplars,nKbase)
        
        #concatenate the base and novel category examples
        Test=Tbase+Tnovel
        random.shuffle(Test)
        Kall=Kbase+Knovel
        
        return Exemplars , Test , Kall , nKbase
    
    def createExamplesTensorData(self,examples):
        """
        Create the examples image and label tensor data
        """
        images=torch.stack(
            [self.dataset[img_idx][0] for img_idx ,_ in examples],dim=0)
        labels=torch.LongTensor([label for _,label in examples])
        return images,labels
    
    def get_iterator(self,epoch=0):
        rand_seed=epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        
        def load_function(iter_idx):
            Exemplars,Test,Kall,nKbase = self.sample_episode()
            Xt,Yt=self.createExamplesTensorData(Test)
            Kall=torch.LongTensor(Kall)
            if len(Exemplars)>0:
                Xe,Ye=self.createExamplesTensorData(Exemplars)
                return Xe,Ye,Xt,Yt,Kall,nKbase
            else:
                return Xt,Yt,Kall,nKbase
            
        tnt_dataset=tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size),load=load_function)
        data_loader=tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(0 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))
        
        return data_loader
    def __call__(self,epoch=0):
        return self.get_iterator(epoch)
    
    def __len__(self):
        return int(self.epoch_size/self.batch_size)