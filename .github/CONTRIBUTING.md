# Contributing to Construction-Hazard-Detection

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to Construction-Hazard-Detection. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Feature Requests](#feature-requests)
   - [Code Contributions](#code-contributions)
3. [Development Setup](#development-setup)
4. [Submitting Changes](#submitting-changes)
5. [Style Guides](#style-guides)
   - [Python Style Guide](#python-style-guide)
   - [Commit Message Guidelines](#commit-message-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by the [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [project email].

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for Construction-Hazard-Detection. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

- **Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/yihong1120/Construction-Hazard-Detection/issues).

- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/yihong1120/Construction-Hazard-Detection/issues/new). Be sure to include:
  - A clear, descriptive title.
  - Steps to reproduce the issue.
  - Expected and actual behavior.
  - Screenshots, logs, or other relevant information.

### Feature Requests

We welcome suggestions for new features or enhancements. To make a feature request, please [open an issue](https://github.com/yihong1120/Construction-Hazard-Detection/issues/new) and include:
- A clear and concise description of the feature.
- The motivation behind the feature.
- Any examples or mockups if applicable.

### Code Contributions

If you want to contribute code to the project, we recommend that you start by reading the [Development Setup](#development-setup) section. Then, you can fork the repository, make your changes, and submit a pull request.

## Development Setup

To set up the development environment:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
   cd Construction-Hazard-Detection
   ```

2. **Set up a virtual environment**:

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install pre-commit hooks**:

   ```bash
   pre-commit install
   ```

5. **Set up environment variables**:

   Create a `.env` file in the root directory with the necessary environment variables (refer to `.env.example` for guidance).

6. **Run the tests**:

   ```bash
   pytest
   ```

## Submitting Changes

When you're ready to submit your changes, follow these steps:

1. **Fork the repository** (if you haven't already).

2. **Create a new branch**:

   ```bash
   git checkout -b your-feature-branch
   ```

3. **Commit your changes** following the [commit message guidelines](#commit-message-guidelines).

4. **Push to your fork**:

   ```bash
   git push origin your-feature-branch
   ```

5. **Open a pull request** on the main repository:
   - Provide a clear description of the changes.
   - Reference related issues if applicable.

6. **Wait for review**:
   - Your pull request will be reviewed, and feedback may be provided.
   - Address any comments or requested changes.

## Style Guides

### Python Style Guide

We follow PEP 8 guidelines for Python code. Some key points:

- Use 4 spaces per indentation level.
- Use `snake_case` for variable and function names.
- Use `CamelCase` for class names.
- Limit lines to 79 characters.
- Use docstrings for documenting functions and classes.

Run `flake8` and `black` to ensure your code adheres to the style guide:

```bash
flake8 .
black .
```

### Commit Message Guidelines

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally.

## Thank You

Thank you for considering contributing to Construction-Hazard-Detection! Your help is essential to keep this project going and improving. We appreciate your time and effort to contribute.

For any questions or additional support, feel free to reach out to [contact email].

Happy coding!
