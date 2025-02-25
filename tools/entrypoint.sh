#!/bin/bash

export VIRTUAL_ENV=/workspace/mostlyai/.venv
export PATH=$VIRTUAL_ENV/bin:$PATH

nohup jupyter lab --port=8888 --ip=0.0.0.0 --allow-root --no-browser --IdentityProvider.token="" --ServerApp.root_dir="." --ServerApp.preferred_dir="./mostlyai/docs/tutorials" &> /jupyter.log &

exec "$@"
