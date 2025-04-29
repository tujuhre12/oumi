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

### Config Schema Definition
- Define JSON Schema for all config types
- Create Pydantic models matching schema
- Use schema for validation and IDE autocompletion

### Config Templates
- Maintain template library for common scenarios
- Parameterize templates for customization
- Version templates alongside code

### Wizard Logic
- Progressive field collection based on dependencies
- Validation at each step
- Helpful error messages and suggestions

### Hardware Detection
- GPU count, type, and memory detection
- Automatic suggestion of optimal training configuration
- Fallback options for resource-constrained environments

### Integration Points
- CLI command via Typer integration
- Python API as importable module

## User Experience Considerations

### New Users
- Start with common recipes as templates
- Guided experience with explanations
- Minimal required parameters
- Smart defaults based on detected hardware

### Experienced Users
- Quick template selection
- Advanced options accessible but not intrusive
- Batch/script support for CI/CD

### Parameter Documentation
- Inline help for each parameter
- Links to relevant documentation
- Examples for complex parameters

## Implementation Phases

### Phase 1: Core Schema and CLI (Focus)
- Define config schemas and validation rules
- Implement basic CLI wizard with templates
- Support train, eval, and infer config types
- Implement manual training type selection

### Phase 2: Enhanced CLI Features
- Add hardware detection for auto-configuration
- Implement resource-aware recommendations
- Add platform-specific job config generation
- Add programmatic API

### Phase 3: Advanced Capabilities
- Config validation and linting tool
- Migration tools for old configs
- Integration with experiment tracking

### Future Consideration
- Web UI for graphical config creation (post Phase 3)

## Evaluation Metrics
- Time saved in config creation
- Reduction in invalid configs
- User satisfaction metrics
- Adoption rate among new users
- Reduction in support requests related to configuration

## Alternatives Considered
- **YAML editors with schema validation**: Less guided, requires schema knowledge
- **Documentation-only approach**: Less interactive, higher cognitive load
- **GUI-first approach**: Less accessible for command-line users, development overhead

## Conclusion
The Config Wizard will significantly improve the onboarding experience for new Oumi users while offering time-saving tools for experts. By providing sensible defaults, hardware-aware recommendations, and interactive guidance, we can reduce configuration errors and help users focus on their ML objectives rather than configuration syntax.