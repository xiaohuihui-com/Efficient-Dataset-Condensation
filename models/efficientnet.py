from efficientnet_pytorch import EfficientNet


def efficientnet(nclass):
    model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    return model