import re

# Read the file
file_path = r'C:\Users\tylor\source\repos\DontDoThat21\QuantraWPF\Quantra\python\train_from_database.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the main function indentation
lines = content.split('\n')
fixed_lines = []
in_main_func = False
base_indent = 4  # Start with 4 spaces for function body
current_indent = base_indent

for i, line in enumerate(lines):
    if line.strip() == 'def main():':
        fixed_lines.append(line)
        in_main_func = True
        current_indent = base_indent
        continue
    
    # If we're in main function and hit another function or end of file
    if in_main_func and line.startswith('if __name__'):
        in_main_func = False
        fixed_lines.append(line)
        continue
    
    # Fix indentation for main function body
    if in_main_func:
        stripped = line.lstrip()
        if not stripped:  # Empty line
            fixed_lines.append('')
            continue
            
        # Calculate proper indentation based on structure
        if stripped.startswith('"""') or stripped.startswith("'''"):
            # Docstring
            fixed_lines.append(' ' * base_indent + stripped)
        elif stripped.startswith('if ') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('with ') or stripped.startswith('try:'):
            fixed_lines.append(' ' * base_indent + stripped)
            current_indent = base_indent + 4
        elif stripped.startswith('except ') or stripped.startswith('elif ') or stripped.startswith('else:') or stripped.startswith('finally:'):
            current_indent = base_indent
            fixed_lines.append(' ' * base_indent + stripped)
            current_indent = base_indent + 4
        elif line.startswith(' ' * (base_indent + 4)):
            # Already properly indented
            fixed_lines.append(line)
        elif line.startswith(' '):
            # Some indentation but not correct - use current level
            fixed_lines.append(' ' * current_indent + stripped)
        else:
            # No indentation - add base level
            fixed_lines.append(' ' * base_indent + stripped)
    else:
        fixed_lines.append(line)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_lines))

print('Fixed indentation successfully!')
