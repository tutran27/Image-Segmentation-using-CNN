import torch.nn as nn

class Encoder_Diagram(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_block=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
    def forward(self, x):
        x=self.conv_block(x)
        return x

class Decoder_Diagram(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv_block=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    def forward(self, x):
        x=self.conv_block(x)
        return x
    
class Segmentation_with_CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels=n_channels
        self.n_classes=n_classes

        self.encoder_1=Encoder_Diagram(n_channels, 64, 3)    # 64
        self.encoder_2=Encoder_Diagram(64, 128, 3)  # 32
        self.encoder_3=Encoder_Diagram(64*2, 64*4, 3)   # 16
        self.encoder_4=Encoder_Diagram(64*4, 64*8, 3)   # 8

        self.encoder_5=Encoder_Diagram(64*8, 64*8, 3)   # 4

        self.decoder_1=Decoder_Diagram(64*8, 64*8, 3)  #8
        self.decoder_2=Decoder_Diagram(64*8, 64*4, 3)  #16
        self.decoder_3=Decoder_Diagram(64*4, 64*2, 3)      #32
        self.decoder_4=Decoder_Diagram(64*2, 64, 3)   #64

        self.out=nn.Sequential(
            nn.Conv2d(64,n_classes, 3, padding='same'),
            nn.BatchNorm2d(n_classes),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
      x=self.encoder_1(x)
      x=self.encoder_2(x)
      x=self.encoder_3(x)
      x=self.encoder_4(x)
      x=self.encoder_5(x)

      x=self.decoder_1(x)
      x=self.decoder_2(x)
      x=self.decoder_3(x)
      x=self.decoder_4(x)

      x=self.out(x)

      return x