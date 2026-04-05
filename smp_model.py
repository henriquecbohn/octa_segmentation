# models/smp_model.py
import segmentation_models_pytorch as smp
import torch

def get_model(encoder_name, encoder_weights="imagenet", in_channels=1, classes=1):

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None  
    )
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for encoder in ["resnet34", "efficientnet-b0"]:
        model = get_model(encoder).to(device)
        x = torch.randn(2, 1, 400, 400).to(device)
        y = torch.sigmoid(model(x))

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{encoder:25s} | params: {total_params:,} | output: {y.shape} | min/max: {y.min():.3f}/{y.max():.3f}")