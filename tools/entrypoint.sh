#!/bin/bash

source /workspace/mostlyai/.venv/bin/activate

python -i -c "code='from mostlyai.sdk import MostlyAI\nmostly = MostlyAI(local=True, local_port=8080)'
print('\n'.join([f'>>> {line}' for line in code.split('\n')]))
exec(code)"

exec "$@"
