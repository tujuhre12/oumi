# Oumi Terminal UI

A terminal-based user interface for Oumi, built with [Textual](https://textual.textualize.io/), to provide an interactive way to explore, configure, and monitor Oumi operations.

## Features

- **Dataset Explorer:** Browse and select datasets by category
- **Model Explorer:** Browse and select models by category
- **Configuration Builder:** Interactively create configurations for training, evaluation, and inference
- **Training Monitor:** View and manage active training runs

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Oumi Terminal UI
python src/oumi_tui.py
```

## Keyboard Shortcuts

- `q`: Quit the application
- `d`: Toggle dark mode
- `F1`: Show help

## Screenshots

*Add screenshots here once the application is fully functional*

## Development

The Oumi Terminal UI is built with Textual, a modern TUI (Text User Interface) framework for Python. It's designed to provide a user-friendly way to interact with Oumi's functionality without needing to manually write configuration files or remember CLI commands.

### Project Structure

- `oumi_tui.py`: Main application code
- `oumi_tui.css`: Textual CSS styling

### Extending the UI

To add new features to the UI, you can:

1. Create new widget classes in `oumi_tui.py`
2. Update the styling in `oumi_tui.css`
3. Add new screens or integrate with additional Oumi functionality

## Future Improvements

- Real-time training progress visualization
- Integration with Oumi's telemetry for live metrics
- Configuration templates and saved configurations
- Wizard-based setup for common workflows
- Direct integration with WandB and TensorBoard
- Support for remote training management

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
