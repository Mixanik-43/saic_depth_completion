from collections import OrderedDict


def submodel_state_dict(state_dict, prefix):
    return OrderedDict({k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)})


def default_set_torch_weights(self, torch_weights):
    tf_weights = []
    for k in torch_weights.keys():
        if k.endswith('.num_batches_tracked'):
            continue
        if len(torch_weights[k].shape) == 4:
            tf_weights.append(torch_weights[k].permute(2, 3, 1, 0))
        else:
            tf_weights.append(torch_weights[k])

    self.set_weights(tf_weights)