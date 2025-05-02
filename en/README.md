# Hetu document

## Build the docs

```shell
# Install dependencies.
pip install -r source/requirements.txt
# Build the docs.
make clean
make html
```

## Open the docs with your browser

```shell
sphinx-autobuild source build/html
```
