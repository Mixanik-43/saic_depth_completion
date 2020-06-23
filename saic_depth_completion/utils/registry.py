class Registry(dict):
    def register(self, name):
        if name in self.keys():
            raise ValueError("Registry already contains such key: {}".format(name))

        def _register(fn):
            self.update({name: fn})
            return fn

        return _register


MODELS = Registry()
TF_MODELS = Registry()
BACKBONES = Registry()
TF_BACKBONES = Registry()

