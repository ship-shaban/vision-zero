# Contributing to Vision Zero Toronto

Thank you for your interest in contributing! This document provides guidelines for contributors.

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Git
- Basic knowledge of Flask and Pandas

### Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/vision-zero.git
   cd vision-zero
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pyarrow  # Recommended for performance
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
   Visit http://localhost:5001

## How to Contribute

### Areas We Need Help

- **Features**: Export functionality, additional filters, advanced search
- **Analysis**: Predictive modeling, hot spot detection, pattern analysis
- **Performance**: Further optimizations, caching strategies
- **UX**: Accessibility (WCAG 2.1), internationalization, dark mode
- **Documentation**: API docs, tutorials, deployment guides

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes**:
   ```bash
   python app.py  # Manual testing
   python validate_implementation.py  # Validation
   ```

4. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description"
   ```

   Good commit messages:
   - ✅ "Add CSV export functionality"
   - ✅ "Fix ward boundary loading issue"
   - ❌ "Fixed stuff"

5. **Push and create a Pull Request**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python
- Follow PEP 8 style guide
- Use type hints
- Maximum line length: 100 characters
- Add docstrings to functions

### JavaScript
- Use ES6+ syntax
- Use `const` and `let`, avoid `var`
- Add comments for complex logic

### HTML/CSS
- Use semantic HTML5
- Follow BEM naming for CSS
- Ensure accessibility (ARIA labels)

## Pull Request Process

Your PR should:
- Have a clear title and description
- Reference related issues (e.g., "Fixes #123")
- Include screenshots for UI changes
- Update documentation if needed
- Be focused on a single feature or fix

## Testing

Run validation scripts before submitting:
```bash
python validate_implementation.py
python validate_against_police_dashboard.py
```

## Questions?

- Open an [issue](https://github.com/ship-shaban/vision-zero/issues) for bug reports or features
- Check existing documentation in [README.md](https://github.com/ship-shaban/vision-zero#readme)

---

Thank you for contributing to making Toronto's streets safer!
