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
A multi-tiered Config Wizard that guides users through config creation:

1. **CLI Wizard**: Interactive command-line interface
2. **Web UI**: Optional browser-based configuration builder [Feedback: This is a good idea, but we should focus on the CLI for now]
3. **Programmatic API**: Python interface for config generation

## Implementation Options

### 1. CLI Wizard  [Feedback: this should take into account the training type: e.g. use fsdp, ddp, peft, lora, qlora. We can start by having the user provide this, but plan for a future "auto" option where it's inferred from the model size and available GPU resources]
```
oumi config create --type train --model llama3-8b
```

#### Approach Options:
- **Template-based**: Start with base templates and prompt for customization
- **Progressive disclosure**: Begin with minimal required fields, offer advanced options
- **Decision-tree**: Guide users based on their prior selections

#### Key Features:
- Interactive parameter setting with validation
- Sensible defaults with explanations
- Preview generated config
- Save to file or output to stdout

### 2. Web UI [Feedback: Let's skip the web UI for now]
A browser-based interface with form fields and validation.

#### Approach Options:
- **Standalone app**: Simple Flask/FastAPI web server
- **Jupyter integration**: Interactive widgets within notebooks
- **Documentation integration**: Embedded in Oumi docs site

#### Key Features:
- Visual form with hierarchical organization
- Real-time validation and suggestions
- Side-by-side YAML preview
- Preset templates for common scenarios

### 3. Programmatic API
Python functions to generate configs programmatically.

```python
from oumi.config import ConfigBuilder

config = (ConfigBuilder()
    .model("meta-llama/Llama-3.1-8B-Instruct")
    .training_type("LoRA")
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

### Integration Points
- CLI command via Typer integration
- Web interface using Gradio or Streamlit
- Python API as importable module

## User Experience Considerations

### New Users
- Start with common recipes as templates
- Guided experience with explanations
- Minimal required parameters

### Experienced Users
- Quick template selection
- Advanced options accessible but not intrusive
- Batch/script support for CI/CD

### Parameter Documentation
- Inline help for each parameter
- Links to relevant documentation
- Examples for complex parameters

## Implementation Phases

### Phase 1: Core Schema and CLI
- Define config schemas and validation rules
- Implement basic CLI wizard with templates
- Support train, eval, and infer config types

### Phase 2: Enhanced Features
- Add platform-specific job config generation
- Implement web UI integration
- Add programmatic API

### Phase 3: Advanced Capabilities
- Config validation and linting tool
- Migration tools for old configs
- Integration with experiment tracking

## Evaluation Metrics
- Time saved in config creation
- Reduction in invalid configs
- User satisfaction metrics
- Adoption rate among new users

## Alternatives Considered
- **YAML editors with schema validation**: Less guided, requires schema knowledge
- **Documentation-only approach**: Less interactive, higher cognitive load
- **GUI-first approach**: Less accessible for command-line users

## Conclusion
The Config Wizard will significantly improve the onboarding experience for new Oumi users while offering time-saving tools for experts. By providing sensible defaults and interactive guidance, we can reduce configuration errors and help users focus on their ML objectives rather than configuration syntax.