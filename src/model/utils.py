import torch
import pathlib, logging
from diffusers import (
    DDIMScheduler, 
    AutoencoderKL, 
    UNet2DConditionModel
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer
)
import yaml


logger = logging.getLogger('model/utils')


_models = {
    'DDIMScheduler': DDIMScheduler,
    'AutoencoderKL': AutoencoderKL,
    'UNet2DConditionModel': UNet2DConditionModel,
    'CLIPTextModel': CLIPTextModel,
    'CLIPTokenizer': CLIPTokenizer
}


def _load_module(module_config: dict, device: str = 'cpu') -> torch.nn.Module:
    print('Start loading model %s' % module_config['classname'])
    model_class = _models[module_config['classname']]
    model = model_class.from_pretrained(module_config['hub'], 
                                        subfolder=module_config['subfolder'])
    try:
        model.to(device)
    except Exception as e:
        print('Error caught while moving to the device: ', e)
    return model


def load_models(config_path: pathlib.Path, device: str = 'cpu') -> dict[torch.nn.Module]:
    result = dict()
    with config_path.open() as config_f:
        config = yaml.safe_load(config_f)

    result['model_name'] = list(config.keys())[0]
    modules = config[result['model_name']]
    for module_name in modules.keys():
        module_config = modules[module_name]
        result[module_name] = _load_module(module_config=module_config, device=device)
    return result