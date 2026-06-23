import re
import os

with open('run.py', 'r') as f:
    content = f.read()

def extract_var(name):
    match = re.search(f"{name}\\s*=\\s*\"\"\"(.*?)\"\"\"", content, re.DOTALL)
    if match:
        return match.group(1)
    return ""

css = extract_var('_CSS')
js = extract_var('_JS')

templates = {
    'index.html': extract_var('INDEX_T'),
    'dashboard.html': extract_var('DASHBOARD_T'),
    'capture.html': extract_var('CAPTURE_T'),
    'history.html': extract_var('HISTORY_T')
}

os.makedirs('app/templates', exist_ok=True)

for name, html in templates.items():
    if not html:
        continue
    
    # Replace the CSS
    html = html.replace('""" + _CSS + """', css)
    
    # Replace the JS
    html = html.replace('""" + _JS + """', js)
    
    with open(f'app/templates/{name}', 'w') as f:
        f.write(html)

print("Templates fully extracted and hydrated.")
