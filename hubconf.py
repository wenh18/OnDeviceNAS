dependencies = ['torch']
from mytimm.models import registry

globals().update(registry._model_entrypoints)
