# Config Wizard Proposal

## Overview
This proposal outlines the implementation of a Config Wizard for Oumi - a guided, interactive tool to help users create valid configuration files for training, evaluation, and inference. The wizard simplifies the complex configuration process for new users while maintaining flexibility for experienced users.

## Problem Statement
Currently, creating Oumi configs requires:
- Deep understanding of YAML structure
- Familiarity with model-specific parameters
- Knowledge of valid/required fields
- Careful copying and modification of existing configs

This creates a steep learning curve for new users and increases the chance of misconfiguration.

## Proposed Solution
A focused Config Wizard approach with two main components:

1. **CLI Wizard**: Interactive command-line interface (primary focus)
2. **Programmatic API**: Python interface for config generation

## Implementation Options

### 1. CLI Wizard
```
oumi config create --type train --model llama3-8b [--training-type lora|qlora|full|auto]
```

#### Training Type Selection
- **Manual Selection**: User explicitly chooses training type (FSDP, DDP, LoRA, QLoRA)
- **Auto Mode**: Training type is inferred based on:
  - Model size (parameters count)
  - Available GPU resources (memory, quantity)
  - User access to model weights (full vs adapter-only)

#### Approach Options:
- **Template-based**: Start with base templates and prompt for customization
- **Progressive disclosure**: Begin with minimal required fields, offer advanced options
- **Decision-tree**: Guide users based on their prior selections

#### Key Features:
- Interactive parameter setting with validation
- Sensible defaults with explanations
- Hardware-aware recommendations for distributed training
- Preview generated config
- Save to file or output to stdout

### 2. Programmatic API
Python functions to generate configs programmatically.

```python
from oumi.config import ConfigBuilder

config = (ConfigBuilder()
    .model("meta-llama/Llama-3.1-8B-Instruct")
    .training_type("auto")  # Will determine best training method based on model size and available resources
    .dataset("yahma/alpaca-cleaned")
    .build())

config.save("my_config.yaml")
```

## Technical Implementation

### Leveraging Existing Schema
- Utilize the existing `oumi.core.configs` module as the foundation
- Work with the defined Pydantic models like `TrainingConfig`, `EvaluationConfig`, and `InferenceConfig`
- Take advantage of OmegaConf for configuration merging and validation
- Use the existing validation logic in the config classes

### Template Library Organization
- Create a dedicated module for wizard templates
- Organize by config type (train, eval, infer) and model family
- Parameterize templates for customization based on user input
- Include metadata about parameter importance and help text

### Wizard CLI Integration
- Add a new command to the existing Typer-based CLI under `oumi.cli`
- Implement as `oumi config create`
- Use prompt toolkit for interactive input
- Integrate with existing config loaders

## Implementation Plan with Milestones

### Milestone 1: MVP (3 weeks)
Focus on creating a minimal viable product that provides immediate value to users.

#### Week 1: Structure & Templates
- [ ] Create a new module structure for the wizard
- [ ] Create template library based on existing recipe configs
- [ ] Implement template selection and modification logic
- [ ] Add parameter metadata (descriptions, importance, etc.)

#### Week 2-3: Basic CLI Wizard
- [ ] Implement CLI command structure with Typer
- [ ] Build interactive prompt flow for training config creation
- [ ] Support basic model selection and training type selection
- [ ] Add config validation using existing config classes
- [ ] Implement output to file with preview

**MVP Deliverable**: Basic CLI wizard that can create training configs from templates.

### Milestone 2: Enhanced CLI (3 weeks)
Build upon the MVP to provide a more comprehensive CLI experience.

#### Week 4-5: Expanded Config Types
- [ ] Add support for evaluation config creation
- [ ] Add support for inference config creation
- [ ] Implement progressive disclosure of parameters
- [ ] Add parameter validation with helpful error messages

#### Week 6: User Experience Improvements
- [ ] Add inline help text for parameters
- [ ] Implement better default suggestion logic
- [ ] Add config preview with syntax highlighting
- [ ] Create integration tests for end-to-end workflow

**Milestone 2 Deliverable**: Full-featured CLI wizard supporting all config types with improved guidance.

### Milestone 3: Hardware-Aware Features (3 weeks)
Add intelligence to the config creation process.

#### Week 7-8: Resource Detection
- [ ] Implement GPU detection (count, type, memory)
- [ ] Create logic for training method selection based on model size
- [ ] Add automatic suggestion of distributed training parameters
- [ ] Support custom resource specification for remote execution

#### Week 9: Auto Configuration
- [ ] Implement "auto" mode for training type selection
- [ ] Add memory requirement estimation for models
- [ ] Create adaptive batch size and gradient accumulation suggestions
- [ ] Add resource verification against config requirements

**Milestone 3 Deliverable**: Smart CLI wizard that can suggest optimal configurations based on hardware.

### Milestone 4: Programmatic API (3 weeks)
Enable programmatic config generation for advanced users and automation.

#### Week 10-11: Core API
- [ ] Design and implement ConfigBuilder API
- [ ] Add method chaining for configuration options
- [ ] Leverage existing config validation
- [ ] Create serialization to YAML and deserialization

#### Week 12: Integration & Documentation
- [ ] Add examples and documentation for API usage
- [ ] Create integration with CLI wizard (import/export)
- [ ] Implement programmatic generation of job configs
- [ ] Add comprehensive API tests

**Milestone 4 Deliverable**: Python API for programmatic config generation fully integrated with the CLI wizard.

### Future Milestones
After the core milestones are completed, additional features can be considered:

1. **Config Migration Tools**: Assist users in upgrading configs between versions
2. **Config Linting and Analysis**: Detect potential issues in existing configs
3. **Job Management Integration**: Connect config creation with job launching
4. **Web UI**: Browser-based interface for visual config creation

## Evaluation Metrics
Progress will be measured against these key metrics:

- **User Adoption**: Percentage of users using the wizard vs. manual config creation
- **Error Reduction**: Decrease in support requests related to configuration issues
- **Time Savings**: Average time to create a valid config (before vs. after)
- **User Satisfaction**: Feedback ratings from users

## Technical Considerations

### Integration with Existing Code
The wizard will integrate with Oumi's existing config infrastructure:

1. **Config Classes**: Using the Pydantic models from `oumi.core.configs` module, including:
   - `TrainingConfig`, `InferenceConfig`, `EvaluationConfig`
   - Parameter classes like `ModelParams`, `DataParams`, `TrainingParams`, etc.

2. **Validation Logic**: Leveraging the `__post_init__` and `finalize_and_validate` methods in the config classes

3. **OmegaConf Integration**: Working with the existing OmegaConf-based config loading and merging

### Command Structure
```
oumi config create train --model <model_name> [--output-file <output_path>]
oumi config create eval --model <model_name> [--output-file <output_path>]
oumi config create infer --model <model_name> [--output-file <output_path>]
```

### Implementation Considerations
- Use `prompt_toolkit` for interactive CLI experience
- Add parameter metadata to existing config classes or in separate files
- Store templates as YAML files with placeholders for customization
- Implement config validation with clear error messages

## Conclusion
The Config Wizard will significantly improve the onboarding experience for new Oumi users while offering time-saving tools for experts. By providing sensible defaults, hardware-aware recommendations, and interactive guidance, we can reduce configuration errors and help users focus on their ML objectives rather than configuration syntax.

Starting with a focused MVP and progressively adding features will ensure we deliver value quickly while building toward a comprehensive solution. This phased approach allows for user feedback to guide later development stages, ensuring the final product meets real user needs.