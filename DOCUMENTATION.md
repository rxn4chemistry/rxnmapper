# Documentation

## Development instructions

Documentation can be automatically generated with Sphinx.

Make sure the dependencies are installed:

```console
pip install -r dev_requirements.txt
```

Afterwards, compile the documentation in the preferred format:

```console
cd docs_source/
sh generate_modules_rst.sh
# html
make html
# latex
make latex
make latexpdf
# GitHub pages
make github
```
