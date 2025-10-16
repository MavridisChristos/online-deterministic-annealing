# venv

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# pip install 

change source files and metadata
python3 -m pip install --upgrade build
python3 -m build (creates dist folder)
python3 -m pip install --upgrade twine
python3 -m twine upload dist/* (API Token saved in local folder)