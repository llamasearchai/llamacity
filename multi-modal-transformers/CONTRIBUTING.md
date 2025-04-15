# Contributing to Multi-Modal Transformers

Thank you for considering contributing to Multi-Modal Transformers! This document outlines the process for contributing to this project, and the guidelines that contributors should follow.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a respectful and welcoming environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/your-username/multi-modal-transformers.git
   cd multi-modal-transformers
   ```
3. **Create a new branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up the development environment**:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Making Changes

1. **Make your changes** in your branch.
2. **Follow the coding style** of the project:
   - We use [Black](https://black.readthedocs.io/) for code formatting.
   - We use [isort](https://pycqa.github.io/isort/) for sorting imports.
   - We use [flake8](https://flake8.pycqa.org/) for linting.
   - These checks are enforced by pre-commit hooks.
3. **Add tests** for your changes:
   - All new features should include tests.
   - All bug fixes should include tests that demonstrate the bug is fixed.
4. **Update documentation** as necessary:
   - Add or update docstrings.
   - Update README or other documentation files if needed.
   - Add examples if appropriate.
5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```
   - Write clear, concise commit messages.
   - Reference issues or pull requests where appropriate.

## Running Tests

Run the tests using pytest:

```bash
pytest
```

For coverage information:

```bash
pytest --cov=src tests/
```

## Building Documentation

To build the documentation locally:

```bash
cd docs
make html
```

Then open `_build/html/index.html` in your browser.

## Submitting a Pull Request

1. **Push your changes** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
2. **Create a pull request** from your fork to the main repository on GitHub.
3. **Describe your changes** in the pull request:
   - What does your PR do?
   - Why is it needed?
   - How has it been tested?
   - Are there any breaking changes?
4. **Address review comments** if any.
5. **Wait for approval** before merging.

## Types of Contributions

### Bug Fixes

If you're fixing a bug:
1. Check if there's an existing issue for the bug. If not, create one.
2. Reference the issue in your pull request.
3. Provide a clear explanation of the bug and how your fix addresses it.

### New Features

If you're adding a new feature:
1. Consider discussing the feature in an issue first to ensure it aligns with the project's goals.
2. Provide a clear explanation of what the feature does and why it's valuable.
3. Include tests and documentation for the new feature.

### Documentation Improvements

Documentation improvements are always welcome! This includes:
- Correcting typos or errors
- Improving clarity or completeness
- Adding examples or tutorials
- Translating documentation

### Code Refactoring

If you're refactoring code:
1. Ensure you don't change the functionality.
2. Explain why the refactoring is beneficial.
3. Ensure all tests pass after the refactoring.

## Style Guidelines

### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
- Use type hints when appropriate.
- Write meaningful docstrings in the [Google style](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).

### Docstrings

- All public classes, methods, and functions should have docstrings.
- Follow the Google style for docstrings:

```python
def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.
    
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    
    Returns:
        bool: The return value. True for success, False otherwise.
    
    Raises:
        ValueError: If param1 is negative.
    
    Examples:
        >>> function_with_types_in_docstring(1, '2')
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be positive")
    return True
```

### Commit Messages

- Use clear, concise messages.
- Start with a short summary line (50 chars or less).
- Optionally followed by a blank line and a more detailed explanation.
- Reference issues or pull requests where appropriate.

## Additional Resources

- [Project Documentation](https://yourusername.github.io/multi-modal-transformers/)
- [Issue Tracker](https://github.com/yourusername/multi-modal-transformers/issues)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)

## Questions?

Feel free to ask questions by opening an issue with the label "question."

Thank you for your contributions!