import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
from model import *
from lsoftmaxloss import LargeMarginInSoftmaxLoss

use_cuda=torch.cuda.is_available()
torch.cuda.set_device(2)
torch.manual_seed(1234)
if use_cuda:
    torch.cuda.manual_seed(1234)

#step 2 
path_fe=''
path_classifier=''

#load pretrain model
fe_model=ConvNet_mobile()
classifier=Classifier(weight_generator_type='attention_based')
pre_train_classifier=torch.load(path_classifier)
fe_model.load(path_fe)

for pname , param in classifier.named_parameters():
    if pname in pre_train_classifier:
        param.data.copy_(pre_train_classifier[pname])
        
        
#load training data
epoch=10
lr=0.1
momentum=0.9
weight_decay=5e-4

dataset_train=Radar_dataset(phase='train')
dataset_test=Radar_dataset(phase='val')

dloader_train=FewShotDataloader(dataset=dataset_train,
                               nKnovel=3,
                               nKbase=-1,
                               nExemplars=5,
                               nTestNovel=4*3,
                               nTestBase=4*3,
                               batch_size=8,
                               num_workers=1,
                               epoch_size=8*500)#8*1000

dloader_test = FewShotDataloader(
    dataset=dataset_test,
    nKnovel=3,
    nKbase=7,
    nExemplars=5, # num training examples per novel category
    nTestNovel=15*8, # num test examples for all the novel categories
    nTestBase=15*8, # num test examples for all the base categories
    batch_size=1,
    num_workers=0,
    epoch_size=200, #2000 num of batches per epoch
)

print(len(dloader_train))
print(len(dloader_test))

def get_labels_train_one_hot(labels_train,num_classes):
    res=[]
    batch_size,num=labels_train.size()
    for i in range(batch_size):
        min_value=torch.min(labels_train[i])
        labels=labels_train[i]-min_value
        one_hot=torch.zeros((num,num_classes))
        for i in range(len(labels)):
            one_hot[i][labels[i]]=1
        res.append(one_hot)
    return torch.cat(res).view(batch_size,num,num_classes)
        
def get_acc(pred,labels):
    _,pred_inds=pred.max(dim=1)
    pred_inds=pred_inds.view(-1)
    labels=labels.view(-1)
    acc=100*pred_inds.eq(labels).float().mean()
    return acc

if not os.path.isdir('results/stage_2_model'):
    os.makedirs('results/stage_2_model')

trace_file=os.path.join('results','trace_file','train_stage_2_trace.txt')
if os.path.isfile(trace_file):
    os.remove(trace_file)
    
if use_cuda:
    fe_model.cuda()
    classifier.cuda()

#optimizer
#optimizer_fe=torch.optim.SGD(fe_model.parameters(),lr=lr,nesterov=True , momentum=momentum,weight_decay=weight_decay)
optimizer_classifier=torch.optim.SGD(classifier.parameters(),lr=lr,nesterov=True , momentum=momentum,weight_decay=weight_decay)
lr_schedule_classifier=torch.optim.lr_scheduler.StepLR(optimizer=optimizer_classifier,gamma=0.5,step_size=25)
criterion=torch.nn.CrossEntropyLoss()
#criterion = LargeMarginInSoftmaxLoss()

print("---- train-stage-2 ----")
best_acc_both=0.0
best_acc_novel=0.0
for ep in range(epoch):
    train_loss=[]
    acc_both=[]
    acc_base=[]
    acc_novel=[]
    print("----epoch: %2d---- "%ep)
    fe_model.train()
    classifier.train()
    
    for batch in tqdm(dloader_train(ep)):
        assert(len(batch)==6) #images_train, labels_train, images_test, labels_test, K, nKbase
        
#         optimizer_fe.zero_grad()
        optimizer_classifier.zero_grad()
        
        train_data=batch[0]
        train_label=batch[1]
        test_data=batch[2]
        test_label=batch[3]
        k_id=batch[4]
        nKbase=batch[5]
        KbaseId=k_id[:,:nKbase[0]]
        labels_train_one_hot=get_labels_train_one_hot(train_label,dloader_train.nKnovel)
        #print('train_data', train_data.size())
        #print('train_label', train_label.size())
        #print('test_data', test_data.size())
        #print('test_label', test_label.size())
        #print('k_id', k_id.size())
        #print('nkbase', nKbase)
        #print('KbasedID', KbaseId)
        #print(labels_train_one_hot.size())
        
        if use_cuda:
            train_data=train_data.cuda()
            train_label=train_label.cuda()
            test_data=test_data.cuda()
            test_label=test_label.cuda()
            k_id=k_id.cuda()
            nKbase=nKbase.cuda()
            KbaseId=KbaseId.cuda()
            labels_train_one_hot=labels_train_one_hot.cuda()
        
        batch_size,nExamples,channels,width,high=train_data.size()
        nTest=test_data.size(1)
        
        train_data=train_data.view(batch_size*nExamples,channels,width,high)
        test_data=test_data.view(batch_size*nTest,channels,width,high)
        
        train_data_embedding=fe_model(train_data)
        test_data_embedding=fe_model(test_data)
        
        pred_result=classifier(features_test=test_data_embedding.view(batch_size,nTest,-1),Kbase_ids=KbaseId,
                               features_train=train_data_embedding.view(batch_size,nExamples,-1),labels_train=labels_train_one_hot)
        #print("pred_result.size",pred_result.size())
        pred_result = pred_result.view(batch_size*nTest,-1)
        #p = nn.Softmax()
        #q = p(pred_result).max(dim=1)
        #print('pre_result', q)
        test_label = test_label.view(batch_size*nTest)
        #print('test_label', test_label)
    
        loss=criterion(pred_result,test_label)
        loss.backward()
#         optimizer_fe.step()
        optimizer_classifier.step()
    
        train_loss.append(float(loss))
        
        accuracy_both=get_acc(pred_result,test_label)
        acc_both.append(float(accuracy_both))
        
        base_ids=torch.nonzero(test_label < nKbase[0]).view(-1)
        novel_ids=torch.nonzero(test_label >= nKbase[0]).view(-1)
        
        pred_base = pred_result[base_ids,:]
        pred_novel =pred_result[novel_ids,:]
        
        accuracy_base=get_acc(pred_base[:,:nKbase[0]],test_label[base_ids])
        accuracy_novel=get_acc(pred_novel[:,nKbase[0]:],(test_label[novel_ids]-nKbase[0]))
        
        acc_base.append(float(accuracy_base))
        acc_novel.append(float(accuracy_novel))
        
    
    lr_schedule_classifier.step()

    #------------------------------------------------------
    #validation stage
    print("----begin validation----")
    fe_model.eval()
    classifier.eval()
    
    val_loss=[]
    val_acc_both=[]
    val_acc_base=[]
    val_acc_novel=[]
    for batch in tqdm(dloader_test(ep)):
        assert(len(batch)==6)
        train_data=batch[0]
        train_label=batch[1]
        test_data=batch[2]
        test_label=batch[3]
        k_id=batch[4]
        nKbase=batch[5]
        KbaseId=k_id[:,:nKbase[0]]
        labels_train_one_hot=get_labels_train_one_hot(train_label,dloader_test.nKnovel)
        #print('train_data', train_data.size())
        #print('train_label', train_label)
        #print('test_data', test_data.size())
        #print('test_label', test_label)
        #print('k_id', k_id.size())
        #print('nkbase', nKbase)
        #print('KbasedID', KbaseId.size())
        #print(labels_train_one_hot.size())
        
        if use_cuda:
            train_data=train_data.cuda()
            train_label=train_label.cuda()
            test_data=test_data.cuda()
            test_label=test_label.cuda()
            k_id=k_id.cuda()
            nKbase=nKbase.cuda()
            KbaseId=KbaseId.cuda()
            labels_train_one_hot=labels_train_one_hot.cuda()
        
        batch_size,nExamples,channels,width,high=train_data.size()
        nTest=test_data.size(1)
        
        train_data=train_data.view(batch_size*nExamples,channels,width,high)
        test_data=test_data.view(batch_size*nTest,channels,width,high)
        
        train_data_embedding=fe_model(train_data)
        test_data_embedding=fe_model(test_data)
        
        pred_result=classifier(features_test=test_data_embedding.view(batch_size,nTest,-1),Kbase_ids=KbaseId,
                               features_train=train_data_embedding.view(batch_size,nExamples,-1),labels_train=labels_train_one_hot)
        #print("pred_result.size",pred_result.size())
        pred_result = pred_result.view(batch_size*nTest,-1)
        p = nn.Softmax()
        q = p(pred_result).max(dim=1)
        #print('pre_result', q)
        test_label = test_label.view(batch_size*nTest)
        #print('test_label', test_label)
        
        loss=criterion(pred_result,test_label)
        val_loss.append(float(loss))
        
        accuracy_both=get_acc(pred_result,test_label)
        val_acc_both.append(float(accuracy_both))
        
        base_ids=torch.nonzero(test_label < nKbase[0]).view(-1)
        novel_ids=torch.nonzero(test_label >= nKbase[0]).view(-1)
        #print('base_ids', base_ids)
        #print('novel_ids', novel_ids)
        #print('test_label', (test_label[novel_ids]-nKbase[0]))
        
        pred_base = pred_result[base_ids,:]
        pred_novel =pred_result[novel_ids,:]
        #print('pred_base', pred_base)
        #print('pred_novel', pred_novel)
        
        accuracy_base=get_acc(pred_base[:,:nKbase[0]],test_label[base_ids])
        accuracy_novel=get_acc(pred_novel[:,nKbase[0]:],(test_label[novel_ids]-nKbase[0]))
        
        val_acc_base.append(float(accuracy_base))
        val_acc_novel.append(float(accuracy_novel))
    avg_loss=np.mean(train_loss)
    avg_acc_both=np.mean(acc_both)
    avg_acc_base=np.mean(acc_base)
    avg_acc_novel=np.mean(acc_novel)
    
    val_avg_loss=np.mean(val_loss)
    val_avg_acc_both=np.mean(val_acc_both)
    val_avg_acc_base=np.mean(val_acc_base)
    val_avg_acc_novel=np.mean(val_acc_novel)
    #print('val_loss', len(val_loss))
    #print('val_both', len(val_acc_both))
    #print('val_base', len(val_acc_base))
    #print('val_novel', len(val_acc_novel))
    
    print("epoch %2d training end : training ---- avg_loss = %.4f , avg_acc_both = %.2f , avg_acc_base = %.2f , avg_acc_novel = %.2f "%(ep,avg_loss,avg_acc_both,avg_acc_base,avg_acc_novel))
    print("epoch %2d training end : validation ---- avg_loss = %.4f , avg_acc_both = %.2f , avg_acc_base = %.2f , avg_acc_novel = %.2f "%(ep,val_avg_loss,val_avg_acc_both,val_avg_acc_base,val_avg_acc_novel))
    with open(trace_file,'a') as f:
        f.write('epoch:{:2d}  training ---- avg_loss:{:.4f} , avg_acc_both:{:.2f} , avg_acc_base:{:.2f} , avg_acc_novel:{:.2f}'.format(ep,avg_loss,avg_acc_both,avg_acc_base,avg_acc_novel))
        f.write('\n')
        f.write('epoch:{:2d}  validation ---- avg_loss:{:.4f} , avg_acc_both:{:.2f} , avg_acc_base:{:.2f} , avg_acc_novel:{:.2f}'.format(ep,val_avg_loss,val_avg_acc_both,val_avg_acc_base,val_avg_acc_novel))
        f.write('\n')
    if best_acc_both<val_avg_acc_both:
        print("produce best both_acc model，saving------")
        
        p1='results/stage_2_model/fe_best_both.pth'
        p2='results/stage_2_model/classifier_best_both.pth'
        m1=fe_model.save(path=p1)
        m2=classifier.save(path=p2)
        best_acc_both=avg_acc_both
        print("successfully saving current best both_acc model----")
    if best_acc_novel<val_avg_acc_novel:
        print("produce best novel_acc model，saving------")
        
        p1='results/stage_2_model/fe_best_novel.pth'
        p2='results/stage_2_model/classifier_best_novel.pth'
        m1=fe_model.save(path=p1)
        m2=classifier.save(path=p2)
        best_acc_novel=avg_acc_novel
        print("succewssfully saving current best novel_acc model----")

