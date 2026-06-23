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
if css:
    css = "<style>\n" + css + "\n</style>"

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
    # Replace """ + _CSS + """ with actual CSS or link
    # But wait, in the templates it looks like: <style>""" + _CSS + """</style>
    html = html.replace('<style>""" + _CSS + """</style>', '{% block styles %}{% endblock %}')
    
    # We should create a base.html if we want, but since they are standalone in run.py,
    # let's just dump the CSS directly into them for now to ensure it works exactly the same, 
    # or write the CSS to a file.
    # Actually, the user wants them in templates/.
    with open(f'app/templates/{name}', 'w') as f:
        f.write(html.replace('""" + _CSS + """', css.replace('<style>\n','').replace('\n</style>','')))

print("Extracted templates.")
