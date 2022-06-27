import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from model import *
from lsoftmaxloss import LargeMarginInSoftmaxLoss

#step 1 =========training Feature Extractor and pretrain cosine-based classifier
use_cuda=torch.cuda.is_available()
torch.cuda.set_device(0)
torch.manual_seed(1234)
if use_cuda:
    torch.cuda.manual_seed(1234)
    
epoch=41
lr=0.1
momentum=0.9
weight_decay=5e-4

dataset_train=Radar_dataset(phase='train')
# dataset_test=MiniImageNet(phase='val')


dloader_train=FewShotDataloader(dataset=dataset_train,
                               nKnovel=0,
                               nKbase=7,
                               nExemplars=0,
                               nTestNovel=0,
                               nTestBase=32,
                               batch_size=8,
                               num_workers=1,
                               epoch_size=8*500)

print(len(dataset_train))

if not os.path.isdir('results/trace_file'):
    os.makedirs('results/trace_file')
    os.makedirs('results/pretrain_model')
    
trace_file=os.path.join('results','trace_file','pre_train_trace.txt')
if os.path.isfile(trace_file):
    os.remove(trace_file)
    
#model
fe_model=ConvNet_mobile()
classifier=Classifier()
if use_cuda:
    fe_model.cuda()
    classifier.cuda()

#optimizer
optimizer_fe=torch.optim.SGD(fe_model.parameters(),lr=lr,nesterov=True , momentum=momentum,weight_decay=weight_decay)
optimizer_classifier=torch.optim.SGD(classifier.parameters(),lr=lr,nesterov=True , momentum=momentum,weight_decay=weight_decay)
lr_schedule_fe=torch.optim.lr_scheduler.StepLR(optimizer=optimizer_fe,gamma=0.5,step_size=25)
lr_schedule_classifier=torch.optim.lr_scheduler.StepLR(optimizer=optimizer_classifier,gamma=0.5,step_size=25)
#criterion=torch.nn.CrossEntropyLoss()
criterion = LargeMarginInSoftmaxLoss()

print("----pre-train----")
for ep in range(epoch):
    train_loss=[]
    print("----epoch: %2d---- "%ep)
    fe_model.train()
    classifier.train()
    
    for batch in tqdm(dloader_train(ep)):
        assert(len(batch)==4)
        
        optimizer_fe.zero_grad()
        optimizer_classifier.zero_grad()
        
        train_data=batch[0]
        train_label=batch[1]
        k_id=batch[2]
        #print('data',train_data.size())
        #print('label',train_label)
        #print('k_id',k_id.size())
        
        if use_cuda:
            train_data=train_data.cuda()
            train_label=train_label.cuda()
            k_id=k_id.cuda()
        
        batch_size,nTestBase,channels,width,high=train_data.size()
        train_data=train_data.view(batch_size*nTestBase,channels,width,high)
        #print('train', train_data.size())
        train_data_embedding=fe_model(train_data)
        #print('embed', train_data_embedding.size())
        pred_result=classifier(train_data_embedding.view(batch_size,nTestBase,-1),k_id)
        #print('pred_result', pred_result.size())
#         print("pred_result.size",pred_result.size())
        loss=criterion(pred_result.view(batch_size*nTestBase,-1),train_label.view(batch_size*nTestBase))
        loss.backward()
        optimizer_fe.step()
        optimizer_classifier.step()
        train_loss.append(float(loss))
    lr_schedule_fe.step()
    lr_schedule_classifier.step()
    
    avg_loss=np.mean(train_loss)
    print("epoch %2d training end : avg_loss = %.4f"%(ep,avg_loss))
    with open(trace_file,'a') as f:
        f.write('epoch:{:2d} training end：avg_loss:{:.4f}'.format(ep,avg_loss))
        f.write('\n')
    if ep==epoch-1:
        p1='results/pretrain_model/femobile_%s.pth'%(str(ep))
        p2='results/pretrain_model/classifiermobile_%s.pth'%(str(ep))
        m1=fe_model.save(path=p1)
        m2=classifier.save(path=p2)
        print("Epoch %2d model successfully saved!"%(ep))
