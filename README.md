## Prerequisites
This program depends on Python3.10.

# Implementation
TBD

# Usage
While designing this repository I kept in mind that end user will be using the following workflow:

![Workflow](misc/Workflow%20of%20StableDiffusion%20Pipeline.drawio.png)

My implementation uses DreamBooth technique to avoid forgetting. So to fine-tune any Text-to-Image StableDiffusion Pipeline you need to provide it with three things:
1. *Instance images.* Commonly 9-12 images of a particular instance you want Pipeline to reproduce.
2. *Instance prompt.* Following the DreamBooth recommendations you must prompt something like "A `instance-name` `class`". So if I want to reproduce the photos of Peter Pomalini I must write something like "A ppomalini person".
3. *Class prompt.* This prompt will be used to generate images of the same class as our instance, so I would recommend prompting "A person."

Then use generated data in training.

## Generate CLI
```
  -h, --help            show this help message and exit
  --input-dir INPUT_DIR
                        Path to the directory containing instance images.
  -c CONFIG, --config CONFIG
                        Config file of the model pipeline.
  -ip INSTANCE_PROMPT, --instance-prompt INSTANCE_PROMPT
                        Prompt to the images of your instance.
  -cp CLASS_PROMPT, --class-prompt CLASS_PROMPT
                        Prompt from which to generate images of the class instances.
  --ratio RATIO         Ratio between class images and instance images. As a default it will, generate 10 class images per one instance image.
  --output-dir OUTPUT_DIR
                        Path to the folder where dataset will be stored. Will be created if not exist.
  -d DEVICE, --device DEVICE
                        Device on which inputs and model itself will be stored.
```
## Train CLI
```
  -h, --help            show this help message and exit
  --data DATA           Path to the dataframe containing data for the training. Data must contain columns: ["image_path", "prompt"]
  -c CONFIG, --config CONFIG
                        Configuration file of the model.
  -e EPOCHS, --epochs EPOCHS
  -d DEVICE, --device DEVICE
                        Device on which model and dataset will be placed on.
  --batch-size BATCH_SIZE
                        Batchsize for the StableDiffusion
  --output OUTPUT       Path to the output folder, where all the weights will be contained.
```
## Evaluate CLI
```
  -h, --help            show this help message and exit
  -p PROMPT, --prompt PROMPT
                        Text prompt to the diffusion model.
  -n NUMBER, --number NUMBER
                        Number of examples to generate.
  --config CONFIG       Path to the config of the generating model.
  -d DEVICE, --device DEVICE
                        Device on which model and inputs will be located.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output of the model.
```
# Example
TBD
# :warning: TODO:
Task name | Progress |
----------|----------|
Write script for evaluation|:white_check_mark:|
Write script for training|:white_check_mark:|
Implement logging|:white_square_button:|
Rethink the way models are being loaded|:white_square_button:|