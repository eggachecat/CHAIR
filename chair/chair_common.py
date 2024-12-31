class DataCollectorFactory:
    
    class DataCollector:
        is_source_constant = True
        keep_last_dim = True
        intervention_types = 'data_collector'
        def __init__(self, component, scope, **kwargs):
            super().__init__(**kwargs)
            self.component = component
            self.scope = scope

        def __call__(self, base, source=None, subspaces=None, model=None):
            self.scope.data_collection[self.component] = model.model.lm_head(base)
            return base

    def __init__(self):
        self.data_collection = {}

    def create(self, component):
        return self.DataCollector(component, scope=self)

    def clear(self):
        self.data_collection = {}

def get_intervention_config(data_collection_factory, name_or_path):
    if 'chatglm3' in name_or_path.lower():
        return  [{
            "component":  f"transformer.encoder.layers.0.input",
            "intervention": data_collection_factory.create(0)
        }] + [
            {
                "component":  f"transformer.encoder.layers.{layer-1}.output",
                "intervention": data_collection_factory.create(layer)
            } for layer in list(range(1,28))
        ], list(range(28))

    if 'llama' in name_or_path.lower():
        intervention_config = [{
            "component":  f"model.layers.0.input",
            "intervention": data_collection_factory.create(0)
        }] + [
            {
                "component":  f"model.layers.{layer-1}.output",
                "intervention": data_collection_factory.create(layer)
            } for layer in list(range(1,33))
        ], list(range(33))

    if 'mistral' in name_or_path.lower():
        return [{
            "component":  f"model.layers.0.input",
            "intervention": data_collection_factory.create(0)
        }] + [
            {
                "component":  f"model.layers.{layer-1}.output",
                "intervention": data_collection_factory.create(layer)
            } for layer in list(range(1,33))
        ], list(range(33))

    raise ValueError('not suppoert model')