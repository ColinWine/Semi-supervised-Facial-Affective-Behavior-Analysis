# =============================================================================
# Author: Xianyuan Liu, xianyuan.liu@sheffield.ac.uk or xianyuan.liu@outlook.com
# =============================================================================

"""Add SELayers to MC3_18, R3D_18, R2plus1D_18"""

from torchvision.models.utils import load_state_dict_from_url

from .video_res3d import (
    BasicBlock,
    BasicStem,
    Conv2Plus1D,
    Conv3DNoTemporal,
    Conv3DSimple,
    R2Plus1dStem,
    VideoResNet,
)
from .video_selayer import get_selayer, SELayerC, SELayerT

model_urls = {
    "r3d_18": "https://download.pytorch.org/models/r3d_18-b3b3357e.pth",
    "mc3_18": "https://download.pytorch.org/models/mc3_18-a90a0ba3.pth",
    "r2plus1d_18": "https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth",
}


def _se_video_resnet_rgb(arch, attention, pretrained=False, progress=True, **kwargs):
    """Add the several SELayers to MC3_18, R3D_18, R2plus1D_18 for RGB input.

    Args:
        arch (string): the name of basic architecture. (Options: ["r3d_18", "mc3_18" and "r2plus1d_18"])
        attention (string): the name of the SELayer.
            (Options: ["SELayerC", "SELayerT", "SELayerCoC", "SELayerMC", "SELayerMAC", "SELayerCT", and "SELayerTC"])
        pretrained (bool): choose if pretrained parameters are used. (Default: False)
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        model (VideoResNet): 3D convolution-based model with SELayers.
    """
    model = VideoResNet(**kwargs)
    temporal_length = 8

    # Add channel-wise SELayer
    if attention in ["SELayerC", "SELayerCoC", "SELayerMC", "SELayerMAC"]:
        se_layer = get_selayer(attention)
        model.layer1._modules["0"].add_module(attention, se_layer(64))
        model.layer1._modules["1"].add_module(attention, se_layer(64))
        model.layer2._modules["0"].add_module(attention, se_layer(128))
        model.layer2._modules["1"].add_module(attention, se_layer(128))
        model.layer3._modules["0"].add_module(attention, se_layer(256))
        model.layer3._modules["1"].add_module(attention, se_layer(256))
        model.layer4._modules["0"].add_module(attention, se_layer(512))
        model.layer4._modules["1"].add_module(attention, se_layer(512))

    # Add temporal-wise SELayer
    elif attention == "SELayerT":
        se_layer = get_selayer(attention)
        model.layer1._modules["0"].add_module(attention, se_layer(temporal_length))
        model.layer1._modules["1"].add_module(attention, se_layer(temporal_length))
        model.layer2._modules["0"].add_module(attention, se_layer(temporal_length // 2))
        model.layer2._modules["1"].add_module(attention, se_layer(temporal_length // 2))
        model.layer3._modules["0"].add_module(attention, se_layer(temporal_length // 4))
        model.layer3._modules["1"].add_module(attention, se_layer(temporal_length // 4))

    # Add channel-temporal-wise SELayer
    elif attention == "SELayerCT":
        model.layer1._modules["0"].add_module(attention + "c", SELayerC(64))
        model.layer1._modules["1"].add_module(attention + "c", SELayerC(64))
        model.layer2._modules["0"].add_module(attention + "c", SELayerC(128))
        model.layer2._modules["1"].add_module(attention + "c", SELayerC(128))
        model.layer3._modules["0"].add_module(attention + "c", SELayerC(256))
        model.layer3._modules["1"].add_module(attention + "c", SELayerC(256))
        model.layer4._modules["0"].add_module(attention + "c", SELayerC(512))
        model.layer4._modules["1"].add_module(attention + "c", SELayerC(512))

        model.layer1._modules["0"].add_module(attention + "t", SELayerT(temporal_length))
        model.layer1._modules["1"].add_module(attention + "t", SELayerT(temporal_length))
        model.layer2._modules["0"].add_module(attention + "t", SELayerT(temporal_length // 2))
        model.layer2._modules["1"].add_module(attention + "t", SELayerT(temporal_length // 2))
        model.layer3._modules["0"].add_module(attention + "t", SELayerT(temporal_length // 4))
        model.layer3._modules["1"].add_module(attention + "t", SELayerT(temporal_length // 4))

    # Add temporal-channel-wise SELayer
    elif attention == "SELayerTC":
        model.layer1._modules["0"].add_module(attention + "t", SELayerT(temporal_length))
        model.layer1._modules["1"].add_module(attention + "t", SELayerT(temporal_length))
        model.layer2._modules["0"].add_module(attention + "t", SELayerT(temporal_length // 2))
        model.layer2._modules["1"].add_module(attention + "t", SELayerT(temporal_length // 2))
        model.layer3._modules["0"].add_module(attention + "t", SELayerT(temporal_length // 4))
        model.layer3._modules["1"].add_module(attention + "t", SELayerT(temporal_length // 4))

        model.layer1._modules["0"].add_module(attention + "c", SELayerC(64))
        model.layer1._modules["1"].add_module(attention + "c", SELayerC(64))
        model.layer2._modules["0"].add_module(attention + "c", SELayerC(128))
        model.layer2._modules["1"].add_module(attention + "c", SELayerC(128))
        model.layer3._modules["0"].add_module(attention + "c", SELayerC(256))
        model.layer3._modules["1"].add_module(attention + "c", SELayerC(256))
        model.layer4._modules["0"].add_module(attention + "c", SELayerC(512))
        model.layer4._modules["1"].add_module(attention + "c", SELayerC(512))

    else:
        raise ValueError("Wrong MODEL.ATTENTION. Current:{}".format(attention))

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def se_r3d_18_rgb(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_rgb(
        "r3d_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] * 4,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def se_mc3_18_rgb(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_rgb(
        "mc3_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
        layers=[2, 2, 2, 2],
        stem=BasicStem,
        **kwargs,
    )


def se_r2plus1d_18_rgb(attention, pretrained=False, progress=True, **kwargs):
    return _se_video_resnet_rgb(
        "r2plus1d_18",
        attention,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[2, 2, 2, 2],
        stem=R2Plus1dStem,
        **kwargs,
    )


def se_r3d(attention, pretrained=False, progress=True):
    """Get R3D_18 models with SELayers for different inputs.

    Args:
        attention (string): the name of the SELayer.
        rgb (bool): choose if RGB model is needed. (Default: False)
        pretrained (bool): choose if pretrained parameters are used. (Default: False)
        progress (bool, optional): whether or not to display a progress bar to stderr. (Default: True)

    Returns:
        model
    """
    return se_r3d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)


def se_mc3(attention, pretrained=False, progress=True):
    """Get MC3_18 models with SELayers for different inputs."""
    return se_mc3_18_rgb(attention=attention, pretrained=pretrained, progress=progress)


def se_r2plus1d(attention, pretrained=False, progress=True):
    """Get R2+1D_18 models with SELayers for different inputs."""
    return se_r2plus1d_18_rgb(attention=attention, pretrained=pretrained, progress=progress)
