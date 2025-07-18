import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class ConvBlock(nn.Module): 
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,1,1)
        self.relu = nn.LeakyReLU(inplace=False)
    
    def forward(self, x): 
        x = self.conv(x)
        x = self.relu(x)
        return x

class ResidualBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, 3)
        self.conv2 = ConvBlock(in_channels, out_channels, 3)
        
        
    def forward(self, x): 
        residual = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)    
        x += residual
        return x 

class UNet(nn.Module): 
    def __init__(self, in_channels=1, out_channels=1, width=32): 
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)
        
        self.encoder1 = nn.Sequential(
            *[ResidualBlock(width, width) for _ in range(3)]
        )
                
        self.encoder2 = nn.Sequential(
            nn.Conv2d(width, width*2, 3, 2, 1),
            *[ResidualBlock(width*2, width*2) for _ in range(3)]
        )
        
        self.encoder3 = nn.Conv2d(width*2, width*4, 3, 2, 1)

            
        self.a_decoder2 = nn.Sequential(
            nn.Conv2d(width*4, width*8, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.a_decoder1 = nn.Sequential(
            *[ResidualBlock(width*2, width*2) for _ in range(2)], 
            nn.Conv2d(width*2, width*4, 3, 1, 1), 
            nn.PixelShuffle(2), 
            ResidualBlock(width, width)
        )
        self.a_decoder0 = nn.Sequential(
            *[ResidualBlock(width, width) for _ in range(2)], 
            nn.Conv2d(width, out_channels, 3, 1, 1)
        )
        
        self.relu = nn.ReLU()

        
    def forward(self,x): 
        
        x = self.intro(x)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        a2 = self.a_decoder2(enc3)
        a2 = a2 + enc2
        a1 = self.a_decoder1(a2)
        a1 = a1 + enc1
        x = self.relu(self.a_decoder0(a1))
        
        return x 
    
class ParameterEstimator(nn.Module): 
    def __init__(self, in_channels=3, width=32): 
        super().__init__()
        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)
        
        self.encoder1 = nn.Sequential(
            *[ResidualBlock(width, width) for _ in range(3)], 
        )
                
        self.encoder2 = nn.Sequential(
            nn.Conv2d(width, width*2, 3, 2, 1),
            *[ResidualBlock(width*2, width*2) for _ in range(3)]
        )
        
        self.encoder3 = nn.Conv2d(width*2, width*4, 3, 2, 1)

            
        self.a_decoder2 = nn.Sequential(
            nn.Conv2d(width*4, width*8, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.a_decoder1 = nn.Sequential(
            *[ResidualBlock(width*2, width*2) for _ in range(2)], 
            nn.Conv2d(width*2, width*4, 3, 1, 1), 
            nn.PixelShuffle(2), 
            ResidualBlock(width, width)
        )
        self.a_decoder0 = nn.Sequential(
            *[ResidualBlock(width, width) for _ in range(2)], 
            nn.Conv2d(width, 3, 3, 1, 1)
        )
        
        self.t_decoder2 = nn.Sequential(
            nn.Conv2d(width*4, width*8, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.t_decoder1 = nn.Sequential(
            *[ResidualBlock(width*2, width*2) for _ in range(2)], 
            nn.Conv2d(width*2, width*4, 3, 1, 1), 
            nn.PixelShuffle(2), 
            ResidualBlock(width, width)
        )
        self.t_decoder0 = nn.Sequential(
            *[ResidualBlock(width, width) for _ in range(2)], 
            nn.Conv2d(width, 1, 3, 1, 1)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.refine = UNet(1,1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self,x,depth): 
        
        # x = torch.cat((x, depth), dim=1)
        x = self.intro(x.clone())
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        a2 = self.a_decoder2(enc3)
        a2 = a2 + enc2
        a1 = self.a_decoder1(a2)
        a1 = a1 + enc1
        a0 = self.sigmoid(self.a_decoder0(a1))
        a = self.global_avg_pool(a0)
        a = self.sigmoid(a)
        a = a.view(a.size(0), -1)
        
        b2 = self.t_decoder2(enc3)
        b2 = b2 + enc2
        b1 = self.t_decoder1(b2)
        b1 = b1 + enc1
        b = self.relu(self.t_decoder0(b1))
        
        t = torch.exp(-b*depth)
        
        refine_t = self.refine(t)
        
        return a, t, refine_t

        
if __name__ == '__main__':
    model = ParameterEstimator(in_channels=3)
    inputs = torch.randn((1,3,32,32))
    
    a, t = model(inputs)
    print(a.shape, t.shape)