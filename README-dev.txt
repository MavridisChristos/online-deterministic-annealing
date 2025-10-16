# venv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# pip install 

change source files and metadata
pip install --upgrade build
build (creates dist folder)
pip install --upgrade twine
twine upload dist/* (PyPI credentials)