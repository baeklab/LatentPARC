import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, pixel_loss=nn.L1Loss(), lambda_pixel=1.0, lambda_perceptual=0.01, device='cpu'):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        loss = self.lambda_pixel * self.pixel_loss(pred, target)

        perc_loss = 0
        for i in range(pred.shape[1]):  # iterate over temperature, pressure, microstructure
            p = pred[:, i:i+1, :, :].repeat(1, 3, 1, 1)   # (B, 1, H, W) -> (B, 3, H, W)
            t = target[:, i:i+1, :, :].repeat(1, 3, 1, 1)
            perc_loss += nn.functional.l1_loss(self.vgg(p), self.vgg(t))

        loss += self.lambda_perceptual * (perc_loss / pred.shape[1])
        return loss

        
class FFTLoss(nn.Module):
    def __init__(self, pixel_loss=nn.L1Loss(), lambda_pixel=1.0, lambda_fft=0.1):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.lambda_pixel = lambda_pixel
        self.lambda_fft = lambda_fft

    def forward(self, pred, target, use_fft=True):
        loss = self.lambda_pixel * self.pixel_loss(pred, target)
        if use_fft:
            pred_fft = torch.fft.rfft2(pred)
            target_fft = torch.fft.rfft2(target)
            loss += self.lambda_fft * nn.functional.l1_loss(
                torch.abs(pred_fft), torch.abs(target_fft)
            )
        return loss

class PhaseFFTLoss(nn.Module): # Includes phase info, not just magnitude
    def __init__(self, pixel_loss=nn.L1Loss(), lambda_pixel=1.0, lambda_fft=0.1):
        super().__init__()
        self.pixel_loss = pixel_loss
        self.lambda_pixel = lambda_pixel
        self.lambda_fft = lambda_fft

    def forward(self, pred, target, use_fft=True):
        loss = self.lambda_pixel * self.pixel_loss(pred, target)
        if use_fft:
            pred_fft = torch.fft.rfft2(pred)
            target_fft = torch.fft.rfft2(target)
            # Instead of (or in addition to) magnitude loss:
            loss += self.lambda_fft * nn.functional.l1_loss(
                torch.view_as_real(pred_fft), 
                torch.view_as_real(target_fft)
            )
        return loss