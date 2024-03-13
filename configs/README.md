# About
Configs are used to build stable diffusion pipeline. It designed in a special way to be flexible: you can build a stable-diffusion pipeline from the various sources and also load them from your local machine!

# Structure
Configuration YAML-files must obey the following structure:
```yaml
model-name:
  module-part-1:
    classname: ClassName1
    storage: local
    path: path/to/model-folder
    train: false
  module-part-2:
    classname: ClassName2
    storage: cloud
    hub: hubname/hubsubfolder
    subfolder: module-part-2-name
    train: true
  ...
```

An example of a common Stable Diffusion Pipeline is located on the path [`configs/default.yaml`](/configs/default.yaml) that instantiates StableDiffusion-2 pipeline:
```yaml
stable-diffusion-2:
  vae:
    classname: AutoencoderKL
    storage: cloud
    hub: stabilityai/stable-diffusion-2
    subfolder: vae
    train: false
  unet:
    classname: UNet2DConditionModel
    storage: cloud
    hub: stabilityai/stable-diffusion-2
    subfolder: unet
    train: true
  scheduler:
    classname: DDIMScheduler
    storage: cloud
    hub: stabilityai/stable-diffusion-2
    subfolder: scheduler
  tokenizer:
    classname: CLIPTokenizer
    storage: cloud
    hub: stabilityai/stable-diffusion-2
    subfolder: tokenizer
  text_encoder: 
    classname: CLIPTextModel
    storage: cloud
    hub: stabilityai/stable-diffusion-2
    subfolder: text_encoder
    train: true
```