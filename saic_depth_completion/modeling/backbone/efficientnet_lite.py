import torch
import geffnet


class EfficientnetLiteEncoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        variant = self.get_variant(config.arch)
        pretrained = config.imagenet
        self.layers = torch.nn.ModuleList([])
        self.feature_channels = []


        efficientet_params = {'efficientnet_lite0': (1.0, 1.0, 224, 0.2),
                              'efficientnet_lite1': (1.0, 1.1, 240, 0.2),
                              'efficientnet_lite2': (1.1, 1.2, 260, 0.3),
                              'efficientnet_lite3': (1.2, 1.4, 280, 0.3),
                              'efficientnet_lite4': (1.4, 1.8, 300, 0.3)}
        arch = self.get_arch(variant)
        if arch not in efficientet_params:
            raise ValueError('unknown efficientnet-lite model: {}'.format(arch))
        model_params = efficientet_params[arch]
        print('loading EfficientnetLiteEncoder', variant, model_params[0], model_params[1], pretrained)
        model = geffnet.gen_efficientnet._gen_efficientnet_lite(variant=variant, channel_multiplier=model_params[0], depth_multiplier=model_params[1],
                                                                pretrained=pretrained).as_sequential()

        self.layers.append(torch.nn.Sequential())
        self.layers.append(torch.nn.Sequential(*model[:3]))

        layer = [model[3]]
        downsamples = 0

        for index in range(4, len(model) - 7):
            for ir_index in range(len(model[index])):
                if model[index][ir_index].conv_dw.stride == (2, 2):
                    downsamples += 1
                    if downsamples > 1:
                        self.layers.append(torch.nn.Sequential(*layer))
                        layer = []
                        downsamples = 1
                layer.append(model[index][ir_index])
        self.layers.append(torch.nn.Sequential(*layer))

        self.classifier = torch.nn.Sequential(
            *model[-7:]
        )

        self.eval()
        x = torch.zeros(1, 3, 224, 224)

        for layer in self.layers:
            x = layer(x)
            self.feature_channels.append(x.shape[1])

    def get_arch(self, model_name):
        return 'efficientnet_lite' + model_name[-1]

    def get_variant(self, model_name):
        if model_name.startswith('tf'):
            return 'tf_efficientnet_lite' + model_name[-1]
        else:
            return 'efficientnet_lite' + model_name[-1]


    def forward(self, input):
        x = input
        features = []

        for index in range(len(self.layers)):
            x = self.layers[index](x)
            features.append(x)

        return features
