pip freeze | grep -E "^[a-zA-Z]" > requirements.in
pip-compile requirements.in
