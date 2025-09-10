import os
import sys

# Add TCPFormer root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tcpformer_root = os.path.abspath(os.path.join(current_dir, '../../../..'))

print("🔍 EXAMINING TCPFORMER NORMALIZATION")
print("=" * 60)

# Check utils/data.py
data_utils_path = os.path.join(tcpformer_root, 'utils/data.py')
if os.path.exists(data_utils_path):
    print("📄 Found utils/data.py - examining denormalize function:")
    with open(data_utils_path, 'r') as f:
        content = f.read()
    
    # Find the denormalize function
    lines = content.split('\n')
    in_denormalize = False
    denorm_lines = []
    
    for line in lines:
        if 'def denormalize' in line:
            in_denormalize = True
            denorm_lines.append(line)
        elif in_denormalize:
            if line.startswith('def ') and 'denormalize' not in line:
                break
            denorm_lines.append(line)
    
    if denorm_lines:
        print("   🔍 denormalize function:")
        for line in denorm_lines:
            print(f"     {line}")
    else:
        print("   ❌ denormalize function not found")

# Check generator_3dhp.py  
generator_path = os.path.join(tcpformer_root, 'data/reader/generator_3dhp.py')
if os.path.exists(generator_path):
    print(f"\n📄 Found generator_3dhp.py:")
    with open(generator_path, 'r') as f:
        content = f.read()
    
    # Look for coordinate processing
    lines = content.split('\n')
    relevant_sections = []
    
    for i, line in enumerate(lines):
        if any(keyword in line.lower() for keyword in ['data_2d', 'poses_2d', 'normalize', 'coord']):
            # Get context
            start = max(0, i-2)
            end = min(len(lines), i+3)
            section = lines[start:end]
            relevant_sections.extend(section)
            relevant_sections.append("---")
    
    if relevant_sections:
        print("   🔍 Coordinate-related code:")
        for line in relevant_sections[:30]:  # Show first 30 lines
            print(f"     {line}")

print("\n💡 KEY QUESTIONS TO ANSWER:")
print("1. Does TCPFormer normalize 2D coordinates to a specific range?")
print("2. What coordinate system does the ground truth use?")
print("3. Are your YOLO coordinates in the same system as ground truth?")
print("4. Does the denormalize function expect specific input ranges?")