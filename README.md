````markdown
# corrca_eeg

This will be my bachelor project on DTU under the Institute of Mathematics and Computer Science. The topic will be Correlated component analysis of EEG as a neural marker of engagement.

## Project structure

The base structure has been created utilizing [cookiecutter](https://github.com/cookiecutter/cookiecutter), using the base template provided from the course [mlops](https://github.com/SkafteNicki/mlops_template) by SkafteNicki (git profile). Any modifications to the project structures will be updated later.

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── corrca_eeg/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
│   ├── original_corrca/      # Copy of original corrca in matlab
│   │   ├──
│   │   └── 
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
