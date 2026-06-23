import os
import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # Fix spaces around minus icon
    content = content.replace(' - ', ' - ')
    
    # Fix minus icon in title tags without spaces
    content = content.replace('Title -', 'Title -')
    content = content.replace('Dashboard -', 'Dashboard -')
    content = content.replace('History -', 'History -')
    content = content.replace('New Scan -', 'New Scan -')
    content = content.replace('The AI Skincare Progress Tracker -', 'The AI Skincare Progress Tracker -')

    # Fix arrow-right in docstrings (with spaces)
    content = content.replace(' -> ', ' -> ')

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed {filepath}")

for root, _, files in os.walk('/home/raman/Documents/ai-skincare-tracker-main'):
    for file in files:
        if file.endswith(('.html', '.py', '.md', '.css', '.js')):
            fix_file(os.path.join(root, file))
