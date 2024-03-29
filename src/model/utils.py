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


def _load_module(module_config: dict, device: str = 'cpu', dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    print('Start loading model %s' % module_config['classname'])
    model_class = _models[module_config['classname']]
    if module_config['storage'] == 'cloud':
        model = model_class.from_pretrained(module_config['hub'], 
                                            subfolder=module_config['subfolder'],
                                            torch_dtype=dtype)
    elif module_config['storage'] == 'local':
        model = model_class.from_pretrained(module_config['path'], torch_dtype=dtype)
    else:
        raise ValueError('Unknown storage type: %s' % module_config['storage'])

    try:
        model.to(device)
    except Exception as e:
        print('Error caught while moving to the device: ', e)
    if module_config.get('train') is not None:
        try:
            model.requires_grad_(module_config['train'])
            print('%s model set requires grad to %s' % 
                  (module_config['classname'], module_config['train']))
        except Exception as e:
            print('Error while setting requires grad.')
    return model


def load_models(config_path: pathlib.Path, device: str = 'cpu', dtype: torch.dtype = torch.bfloat16) -> dict[torch.nn.Module]:
    result = dict()
    with config_path.open() as config_f:
        config = yaml.safe_load(config_f)

    result['model_name'] = list(config.keys())[0]
    modules = config[result['model_name']]
    for module_name in modules.keys():
        module_config = modules[module_name]
        result[module_name] = _load_module(module_config=module_config, device=device, dtype=dtype)
    return result


def generate_gaussian_noise(shape: tuple, device: str, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(
        size=(1, *shape),
        generator=generator
    ).to(device)


def get_target(scheduler, noise, latents, timesteps):
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        return noise
    if pred_type == "v_prediction":
        return scheduler.get_velocity(latents, noise, timesteps)
    raise ValueError(f"Unknown prediction type {pred_type}")


def prior_preserving_loss(model_pred, target, weight) -> torch.Tensor:
    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
    target, target_prior = torch.chunk(target, 2, dim=0)

    loss = torch.functional.F.mse_loss(
        model_pred.float(), target.float(), reduction="mean"
    )
    prior_loss = torch.functional.F.mse_loss(
        model_pred_prior.float(), target_prior.float(), reduction="mean"
    )
    return loss + weight * prior_loss