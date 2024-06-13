import torch
import torch.nn as nn

class conv_deconv3(nn.Module):

    def __init__(self):
        super(conv_deconv3,self).__init__()

        #Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv1_drop = nn.Dropout2d()
        self.non_linearity_1 = nn.ReLU()

        #Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_3 = nn.ReLU()

        #De Convolution 1
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv1.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_4 = nn.ReLU()

        #Max UnPool 1
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5)
        nn.init.kaiming_normal_(self.deconv2.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_5 = nn.ReLU()

        #Max UnPool 2
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv3.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_6 = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.non_linearity_1(out)
        out = self.conv1_drop(out)
        size1 = out.size()
        out, indices1 = self.maxpool1(out)
        out = self.conv2(out)
        out = self.non_linearity_2(out)
        size2 = out.size()
        out, indices2 = self.maxpool2(out)
        out = self.conv3(out)
        out = self.non_linearity_3(out) 

        out = self.deconv1(out)
        out = self.non_linearity_4(out)
        out = self.maxunpool1(out,indices2,size2)
        out = self.deconv2(out)
        out = self.non_linearity_5(out)
        out = self.maxunpool2(out,indices1,size1)
        out = self.deconv3(out)
        out = self.non_linearity_6(out)
        return(out)


class conv_deconv_128(nn.Module):
    """
    the number appended to the model represent the max number of channels
    """

    def __init__(self):
        super(conv_deconv_128,self).__init__()

        #Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv1_drop = nn.Dropout2d()
        self.non_linearity_1 = nn.ReLU()

        #Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_3 = nn.ReLU()

        #De Convolution 1
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv1.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_4 = nn.ReLU()

        #Max UnPool 1
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5)
        nn.init.kaiming_normal_(self.deconv2.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_5 = nn.ReLU()

        #Max UnPool 2
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv3.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_6 = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.non_linearity_1(out)
        out = self.conv1_drop(out)
        size1 = out.size()
        out, indices1 = self.maxpool1(out)
        out = self.conv2(out)
        out = self.non_linearity_2(out)
        size2 = out.size()
        out, indices2 = self.maxpool2(out)
        out = self.conv3(out)
        out = self.non_linearity_3(out) 

        out = self.deconv1(out)
        out = self.non_linearity_4(out)
        out = self.maxunpool1(out,indices2,size2)
        out = self.deconv2(out)
        out = self.non_linearity_5(out)
        out = self.maxunpool2(out,indices1,size1)
        out = self.deconv3(out)
        out = self.non_linearity_6(out)
        return(out)


##########################################################
############ THIS IS NOT WORKING !!!!!!! #################
##########################################################
class conv_deconv_4layers(nn.Module):

    def __init__(self):
        super(conv_deconv_4layers,self).__init__()

        #Convolution 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv1_drop = nn.Dropout2d()
        self.non_linearity_1 = nn.ReLU()

        #Max Pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_3 = nn.ReLU()

        #Max Pool 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, return_indices=True)

        #Convolution 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_4 = nn.ReLU()

        #De Convolution 1
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv1.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_5 = nn.ReLU()

        #Max UnPool 1
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5)
        nn.init.kaiming_normal_(self.deconv2.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_6 = nn.ReLU()

        #Max UnPool 2
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv3.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_7 = nn.ReLU()

        #Max UnPool 3
        self.maxunpool3 = nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 4
        self.deconv4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3)
        nn.init.kaiming_normal_(self.deconv4.weight, mode='fan_in', nonlinearity='relu')
        self.non_linearity_8 = nn.ReLU()

    def forward(self,x):
        out = self.conv1(x)
        out = self.non_linearity_1(out)
        out = self.conv1_drop(out)
        size1 = out.size()
        out, indices1 = self.maxpool1(out)

        out = self.conv2(out)
        out = self.non_linearity_2(out)
        size2 = out.size()
        out, indices2 = self.maxpool2(out)

        out = self.conv3(out)
        out = self.non_linearity_3(out)
        size3 = out.size()
        out, indices3 = self.maxpool3(out)

        out = self.conv4(out)
        out = self.non_linearity_4(out) 



        out = self.deconv1(out)
        out = self.non_linearity_5(out)
        out = self.maxunpool1(out,indices3,size3)
        
        out = self.deconv2(out)
        out = self.non_linearity_6(out)
        out = self.maxunpool2(out,indices2,size2)
        
        out = self.deconv3(out)
        out = self.non_linearity_7(out)
        out = self.maxunpool3(out,indices1,size1)

        out = self.deconv4(out)
        out = self.non_linearity_8(out)

        return(out)