import sys

with open('run.py', 'r') as f:
    code = f.read()

# Disable app.run to prevent it from blocking
code = code.replace("app.run(", "pass # app.run(")

with open('run_temp.py', 'w') as f:
    f.write(code)

import run_temp

import os
os.makedirs('app/templates', exist_ok=True)

with open('app/templates/index.html', 'w') as f:
    f.write(run_temp.INDEX_T)
with open('app/templates/dashboard.html', 'w') as f:
    f.write(run_temp.DASHBOARD_T)
with open('app/templates/capture.html', 'w') as f:
    f.write(run_temp.CAPTURE_T)
with open('app/templates/history.html', 'w') as f:
    f.write(run_temp.HISTORY_T)

print("Extraction successful using import!")
