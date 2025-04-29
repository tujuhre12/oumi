# Customizing Oumi

Often times when using a new framework, you may find that something you'd like to use is
missing. We always welcome [contributions](/development/contributing), but we also understand that sometimes it's
simpler to prototype changes locally. Whether you want to quickly experiment with new
features, test out different implementation approaches, or iterate rapidly on your ideas
without impacting the main codebase, Oumi provides a simple way to support
local customizations without any additional installations.

## Config Wizard

Oumi provides a Config Wizard to help you generate, validate, and optimize configurations for training, evaluation, and inference. The wizard offers both a CLI and a programmatic API.

### Creating Configurations

#### Interactive CLI Wizard

The CLI Wizard provides an interactive way to create configuration files with adaptive recommendations based on your hardware:

```bash
# Create a training configuration
oumi config create train --model meta-llama/Llama-3.1-8B-Instruct

# Create a training configuration with a specific training type
oumi config create train --model meta-llama/Llama-3.1-8B-Instruct --training-type lora

# Create an evaluation configuration
oumi config create eval --model meta-llama/Llama-3.1-8B-Instruct

# Create an inference configuration and save to file
oumi config create infer --model meta-llama/Llama-3.1-8B-Instruct --output my_config.yaml
```

When creating a configuration, the wizard will:
1. Detect your available GPU resources (count, memory, model)
2. Estimate the model size based on its name
3. Calculate memory requirements for different training methods
4. Recommend the best training method if "auto" is selected
5. Suggest appropriate batch sizes and gradient accumulation steps
6. Enable or disable distributed training features like FSDP based on hardware

After setting up key parameters, the wizard will display a preview of the generated configuration and allow you to save it to a file.

#### Programmatic API

For more advanced use cases or batch configuration generation, you can use the programmatic API:

```python
from oumi.utils.wizard import create_train_config, create_eval_config, create_infer_config

# Create a training configuration with LoRA
config = create_train_config(
    model="meta-llama/Llama-3.1-8B-Instruct",
    training_type="lora",
    dataset="yahma/alpaca-cleaned",
    description="My custom training config"
)

# Save to YAML file
config.save("my_train_config.yaml")

# Create an evaluation configuration
eval_config = create_eval_config(
    model="meta-llama/Llama-3.1-8B-Instruct",
    description="My evaluation config"
)
eval_config.save("my_eval_config.yaml")

# Create an inference configuration
infer_config = create_infer_config(
    model="meta-llama/Llama-3.1-8B-Instruct"
)
infer_config.save("my_infer_config.yaml")
```

For more granular control, you can access the full ConfigBuilder API:

```python
from oumi.utils.wizard.config_builder import ConfigBuilder, ConfigType, TrainingMethodType

# Create a custom training config
builder = ConfigBuilder(ConfigType.TRAIN)
builder.set_model("meta-llama/Llama-3.1-8B-Instruct")
builder.set_training_type("lora")
builder.set_dataset("yahma/alpaca-cleaned")
builder.set_description("Custom training configuration")

# Access and modify the underlying config directly if needed
config = builder.build()
config.training.learning_rate = 5e-5

# Export to YAML
yaml_str = builder.build_yaml()
print(yaml_str)

# Save to file
builder.save("custom_config.yaml")
```

### Validating and Analyzing Configurations

The wizard includes powerful tools for linting and analyzing existing configuration files.

#### Linting Configurations

Lint your configs to catch errors and potential issues before running them:

```bash
# Basic linting (errors, warnings, suggestions)
oumi config lint path/to/config.yaml
```

The linter will check for:
- Missing required parameters
- Incompatible settings (e.g., QLoRA with FSDP)
- Hardware compatibility issues
- Performance pitfalls

Sample output:
```
┏━━━━━━━━━━━┓
┃  Linting  ┃
┗━━━━━━━━━━━┛
┃  Oumi Config Linter  ┃

[bold]Linting Results[/bold]
├── [bold red]Errors[/bold red] (must be fixed)
│   ├── [red]Missing model name[/red]
│   └── [red]Batch size must be at least 1[/red]
└── [bold yellow]Warnings[/bold yellow] (should be reviewed)
    └── [yellow]Large model (70B parameters) without FSDP enabled may cause OOM errors[/yellow]
```

#### Detailed Analysis

Get comprehensive analysis of your configs with hardware-specific insights:

```bash
# Detailed analysis with hardware recommendations
oumi config lint --detailed path/to/config.yaml

# Alternative command for the same analysis
oumi config analyze path/to/config.yaml
```

The analyzer provides detailed information about:
- Model size and resource requirements
- Memory usage estimates for your configuration
- Compatibility with your available hardware
- Performance characteristics (batch size, effective batch size)
- Optimization recommendations

Sample output:
```
┏━━━━━━━━━━━┓
┃  Analysis  ┃
┗━━━━━━━━━━━┛
┃  Oumi Config Analyzer  ┃

...

┏━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Configuration Overview ┃
┗━━━━━━━━━━━━━━━━━━━━━━━┛
┃ Property        ┃ Value                              ┃
│ config_type     │ TRAIN                              │
│ model_name      │ meta-llama/Llama-3.1-70B-Instruct  │
│ model_size_bill │ 70.0                               │
│ training_type   │ qlora                              │
│ dataset         │ yahma/alpaca-cleaned               │

┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Performance Characteristics ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━┛
┃ Metric                    ┃ Value             ┃
│ per_device_batch_size     │ 2                 │
│ gradient_accumulation_steps │ 16              │
│ effective_batch_size      │ 32                │
│ learning_rate             │ 0.0001            │
│ epochs                    │ 3                 │
│ gradient_checkpointing    │ False             │

┏━━━━━━━━━━━━━━━━━━━━┓
┃ Resource Requirements ┃
┗━━━━━━━━━━━━━━━━━━━━┛
┃ Resource                    ┃ Value             ┃
│ estimated_memory_per_gpu_gb │ 0.07 GB           │
│ estimated_total_memory_gb   │ 0.07 GB           │
│ available_gpu_memory_gb     │ 80.00 GB          │
│ gpu_count                   │ 2                 │
│ has_sufficient_memory       │ True              │

[bold]Recommendations[/bold]
└── [green]Consider increasing batch size for better training throughput[/green]
```

#### Programmatic Validation

You can also perform linting and analysis programmatically:

```python
from oumi.utils.wizard.config_builder import ConfigBuilder

# Create or load a configuration
builder = ConfigBuilder(ConfigType.TRAIN)
builder.set_model("meta-llama/Llama-3.1-8B-Instruct")
builder.set_training_type("qlora")

# Validate the configuration
is_valid = builder.validate()  # Returns True/False

# Get detailed linting results
issues = builder.lint()
if issues["errors"]:
    print("Configuration has errors:")
    for error in issues["errors"]:
        print(f"- {error}")

# Get performance analysis
analysis = builder.analyze()
print(f"Estimated memory usage: {analysis['resources']['estimated_memory_per_gpu_gb']:.2f} GB")
for recommendation in analysis["recommendations"]:
    print(f"- {recommendation}")
```

### Hardware-Aware Features

The Config Wizard makes intelligent recommendations based on your available hardware:

#### Model Size Detection

The wizard automatically estimates the size of a model based on its name:
- Pattern matching for common formats (e.g., "llama-3-70b" → 70B parameters)
- Known model database with size information
- Support for all major model families (Llama, Phi, Gemma, Mistral, etc.)

#### GPU Resource Detection

The wizard detects and analyzes your GPU resources:
- Number of available GPUs
- Total and available memory
- Support for calculating multi-GPU setups
- Specific GPU model detection

#### Memory Requirement Estimation

For each training method (Full, LoRA, QLoRA), the wizard estimates:
- Per-GPU memory requirements
- Total system memory requirements
- Minimum hardware needed for training
- Reasonable batch sizes and gradient accumulation steps

#### Performance Optimization

For optimal performance, the wizard recommends:
- Appropriate training methods based on hardware constraints
- Batch size and gradient accumulation settings
- Whether to use distributed training features (FSDP, etc.)
- Learning rate adjustments based on effective batch size

### Supported Training Methods

The wizard supports the following training methods:

- **Full Fine-tuning**: Trains all parameters of the model, requiring more memory but potentially yielding better results.
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning that only updates a small number of adapter parameters.
- **QLoRA (Quantized LoRA)**: Combines 4-bit quantization with LoRA for even more memory-efficient training.
- **Auto**: Automatically selects the best method based on model size and hardware resources.

## The Oumi Registry

We support customization via the {py:class}`oumi.core.registry.Registry`.

You can easily register classes that are then loaded as if they're a native part of the
Oumi framework.

See the diagram below for how we load your custom code:

```{mermaid}
%%{init: {'theme': 'base', 'themeVariables': { 'background': '#f5f5f5'}}}%%
graph LR
    %% Oumi Framework
    FR[Oumi] --> |Read OUMI_EXTRA_DEPS_FILE| RQ[requirements.txt]

    %% Load Custom Files
    RQ --> |Import File| CF1[Custom Class 1 File]
    RQ --> |Import File| CF2[Custom Class 2 File]
    RQ --> |Import File| CF3[        ...        ]

    %% Style for core workflow
    style FR fill:#1565c0,color:#ffffff
    style RQ fill:#1565c0,color:#ffffff
    style CF1 fill:#1565c0,color:#ffffff
    style CF1 fill:#1565c0,color:#ffffff
    style CF2 fill:#1565c0,color:#ffffff
    style CF3 fill:#1565c0,color:#ffffff
```

You can register your custom code in two simple steps:
1. Writing a custom Model, Dataset, Cloud, etc
2. Creating a `requirements.txt` file so your code is available via the CLI

## Writing Custom Classes

### Custom Models

You can easily customize models with oumi if something isn't supported out of the box.
We often hear requests for custom loss and custom model architectures: both are simple
to implement via a custom model.

Check out our guide for an in-depth explanation: {doc}`/resources/models/custom_models`

```{note}
Don't forget to decorate your class with

`@registry.register(..., registry.RegistryType.MODEL)`!
```

### Custom Datasets

Custom datasets are a great way to handle unique dataset formats that Oumi may not yet
support.

See the following snippets for examples of custom datasets:
- [Custom SFT Dataset](/resources/datasets/sft_datasets.md#adding-a-new-sft-dataset)
- [Custom Pre-training Dataset](/resources/datasets/pretraining_datasets.md#adding-a-new-pre-training-dataset)
- [Custom Preference Tuning Dataset](/resources/datasets/preference_datasets.md#creating-custom-preference-dataset)
- [Custom Vision-Language Dataset](/resources/datasets/vl_sft_datasets.md#adding-a-new-vl-sft-dataset)
- [Custom Numpy Dataset](sample-custom-numpy-dataset)

```{note}
Don't forget to decorate your class with `@register_dataset(...)`!
```

### Custom Clouds/Clusters

Adding a custom cloud is perfect for handling local clusters not hosted by major cloud
providers.

For example, we wrote a custom cloud for the Polaris platform. Our research team used
this cloud to schedule jobs seamlessly on a remote super computer.

Take a look at our [custom cluster tutorial here](/user_guides/launch/custom_cluster).

```{note}
Don't forget to decorate your class with `@register_cloud_builder(...)`!
```

### Custom Judge Configs

For quick reference, you can register custom judge configs

You can find [examples of custom judge configs here](https://github.com/oumi-ai/oumi/blob/main/src/oumi/judges/judge_court.py).

```{note}
Don't forget to decorate your function with `@register_judge(...)`!
```


## Enable Your Classes for the CLI

If you're using Oumi as a python library, your custom classes will work out of the box!
However, to use your custom classes from the Oumi CLI, you need to tell Oumi which files
to load when initializing our Registry.

To do this, you must first create a `requirements.txt` file. This file has a simple
structure: each line must be an absolute filepath to the file with your custom class /
function (that you specified with the `@register...` decorator).

For example, if you created two custom classes in files `/path/to/custom_cloud.py` and
`/another/path/to/custom_model.py`, your `requirements.txt` files should look like:

```
/path/to/custom_cloud.py
/another/path/to/custom_model.py
```

With your `requirements.txt` file created, you simply need to set the
`OUMI_EXTRA_DEPS_FILE` environment variable to the location of your file, and you're good to go!

``` {code-block} shell
export OUMI_EXTRA_DEPS_FILE=/another/path/requirements.txt
```

## See Also

- {py:class}`oumi.core.models.BaseModel` - Base class for all Oumi models
- {py:class}`oumi.core.registry.Registry` - Model registration system
- {py:class}`oumi.core.configs.params.model_params.ModelParams` - Base parameters class for models
- {gh}`➿ Training CNN on Custom Dataset <notebooks/Oumi - Training CNN on Custom Dataset.ipynb>` - Sample Jupyter notebook using {py:class}`oumi.models.CNNClassifier` and [Custom Numpy Dataset](sample-custom-numpy-dataset).
