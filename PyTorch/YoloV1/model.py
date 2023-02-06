import torch 

#Yolo Architecture directly copied from the paper

architecture_config =[   #One Tuple represents (kernel_size, filters for output, strides, padding)
    (7,64,2,3),
    "MP",
    (3,192,1,1),
    (1,128,1,0),
    (1,256,1,0),
    (3,512,1,1),
    "MP",
    [(1,256,1,0), (3,512,1,1), 4], #two layers which are to be repeated 4 times in this sequence
    (1,512,1,0),
    (3,1024,1,1),
    "MP",
    [(1,512,1,0), (3,1024,1,1), 2],
    (3,1024,1,1),
    (3,1024,2,1),
    (3,1024,1,1),
    (3,1024,1,1)
]

class CNN_Layers(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN_Layers, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) #bias = False because we will be using BatchNorm
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.leakyrelu = torch.nn.LeakyReLU(0.1)

    def forward(self,x):
        x=self.conv
        x=self.batchnorm(x)
        x=self.leakyrelu(x)
        return x

class Yolo(torch.nn.Module):
    def __init__(self, in_channels=3, **kwargs):   #for rgb images
        super(Yolo, self).__init__()  
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)

    def forward(self,x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1)) #start_dim = 1 because we don't want to flatten the number of examples

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels 
        for x in architecture:
            if type(x) == tuple:
                layers += [CNN_Layers(in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
            elif(type(x)==str):
                layers+=[torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            elif(type(x)==list):
                conv1=x[0]
                conv2=x[1] 
                num_repeating = x[2] #integer
                for j in range(num_repeating):
                    layers+=[CNN_Layers(in_channels, out_channels=conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers+=[CNN_Layers(in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels=conv2[1] #the output of the previous layer will be the input of the current layer hence the out_channels of previous conv layer is the input_channel of the current one
        return torch.nn.Sequential(*layers) #Unpacks the list and converts the layers into torch Sequential type

    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return torch.nn.Sequential(
                                   torch.nn.Flatten(),
                                   torch.nn.Linear(1024*S*S, 496), #In the original paper it is 4096 instead of 496, but we are keeping that number to prevent VRAM for being filled
                                   torch.nn.Dropout(p=0.5),
                                   torch.nn.LeakyReLU(0.1),
                                   torch.nn.Linear(496, S*S*(C+B*5))
                                  )

def test(S=7, B=2, C=3):
    model = Yolo(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((3, 224, 224))
    print(model(x))

test()

