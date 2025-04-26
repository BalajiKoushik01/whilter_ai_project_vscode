# Contributing to Vehicle Detection System

Thank you for considering contributing to the Vehicle Detection System! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- Check if the bug has already been reported in the [Issues](https://github.com/username/vehicle_detection/issues)
- Use the bug report template to create a new issue if needed
- Include detailed steps to reproduce the bug
- Include information about your environment (OS, Python version, etc.)

### Suggesting Features

- Check if the feature has already been suggested in the [Issues](https://github.com/username/vehicle_detection/issues)
- Use the feature request template to create a new issue if needed
- Explain why this feature would be useful to most users

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request using the provided template

## Development Setup

1. Clone the repository
   ```
   git clone https://github.com/username/vehicle_detection.git
   cd vehicle_detection
   ```

2. Create a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   pip install -e .
   ```

## Coding Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Add docstrings to all functions, classes, and modules
- Include type hints for function parameters and return values
- Write unit tests for new functionality
- Keep functions small and focused on a single task

## Testing

- Run tests before submitting a pull request
  ```
  pytest
  ```

- Ensure your code passes linting
  ```
  flake8
  ```

## Documentation

- Update documentation for any changes to the API
- Include examples for new features
- Keep the README.md up to date

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).