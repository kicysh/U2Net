from skimage import io, transform

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
from PIL import Image
import os

from u2net.data_loader import Rescale
from u2net.data_loader import RescaleT
from u2net.data_loader import RandomCrop
from u2net.data_loader import ToTensor
from u2net.data_loader import ToTensorLab
from u2net.data_loader import SalObjDataset

from u2net.model import U2NET
from u2net.model import U2NETP

# ------- 1. define loss function --------

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    
    bce_loss = nn.BCELoss(size_average=True)
    
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

def set_directory_of_training_dataset(
        data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep),
        tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep),
        tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep),
        image_ext = '.jpg',
        label_ext = '.png',
        model_dir = "/path/to/models/",
        batch_size_train = 12,
    ):
    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
    
    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]
    
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]
    
        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)
    
    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")
    
    train_num = len(tra_img_name_list)
    
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)
        
    return salobj_dataloader


def training_process(
        net,
        salobj_dataloader,
        optimizer = None,
        save_model_dir = "/path/to/model_weights/",
        is_use_cuda = torch.cuda.is_available(),
        epoch_num = 100000,
        batch_size_train = 12,
        batch_size_val = 1,
        train_num = 0,
        val_num = 0,
        ite_num = 0,
        running_loss = 0.0,
        running_tar_loss = 0.0,
        ite_num4val = 0,
        save_frq = 2000,
):
    if is_use_cuda:
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) if optimizer == None else optimizer
       
    for epoch in range(0, epoch_num):
        net.train()
    
        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
    
            inputs, labels = data['image'], data['label']
    
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
    
            # wrap them in Variable
            if is_use_cuda:
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
    
            # y zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
    
            loss.backward()
            optimizer.step()
    
            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()
    
            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
    
            if ite_num % save_frq == 0:
    
                torch.save(net.state_dict(), save_model_dir+"model_weight_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
    
    return net


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


def predict_process(
    net,
    is_use_cuda = torch.cuda.is_available(),
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images'),
    prediction_dir = "/path/to/pred_dir/",
    model_dir = '/path/to/saved_model.pth',
    ):
    img_name_list = glob.glob(image_dir + os.sep + '*')
    
    print(img_name_list)

    # --------- dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    if is_use_cuda:
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if is_use_cuda:
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)

        del d1,d2,d3,d4,d5,d6,d7
