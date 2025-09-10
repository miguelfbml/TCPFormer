import os
import sys
import numpy as np

# Add TCPFormer root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tcpformer_root = os.path.abspath(os.path.join(current_dir, '../../../..'))
sys.path.insert(0, tcpformer_root)

print(f"TCPFormer root: {tcpformer_root}")
print("=" * 60)

# 1. Find and analyze data loading files
data_files = []
for root, dirs, files in os.walk(tcpformer_root):
    for file in files:
        if (file.endswith('.py') and 
            ('data' in file.lower() or 'loader' in file.lower() or '3dhp' in file.lower()) and
            'preprocess' not in root):
            data_files.append(os.path.join(root, file))

print("🔍 FOUND DATA-RELATED FILES:")
for file in data_files[:10]:  # Show first 10
    rel_path = os.path.relpath(file, tcpformer_root)
    print(f"  - {rel_path}")

print("\n📁 ANALYZING KEY DATA FILES:")

# 2. Look for MPI3DHP specific data loader
mpi3dhp_files = [f for f in data_files if '3dhp' in f.lower()]
if mpi3dhp_files:
    print(f"\n🎯 Found MPI3DHP files: {len(mpi3dhp_files)}")
    for file in mpi3dhp_files:
        print(f"  - {os.path.relpath(file, tcpformer_root)}")
        
        # Read and analyze the file
        try:
            with open(file, 'r') as f:
                content = f.read()
            
            print(f"\n📄 ANALYZING: {os.path.basename(file)}")
            
            # Look for key processing functions
            key_terms = ['normalize', 'data_2d', 'transform', 'preprocess', 'camera', 'coord']
            for term in key_terms:
                if term in content.lower():
                    lines = content.split('\n')
                    relevant_lines = [line for line in lines if term in line.lower() and not line.strip().startswith('#')]
                    if relevant_lines:
                        print(f"  🔍 '{term}' related code:")
                        for line in relevant_lines[:3]:  # Show first 3 matches
                            print(f"    {line.strip()}")
            
        except Exception as e:
            print(f"    ❌ Error reading {file}: {e}")

# 3. Look for config files
print("\n⚙️ LOOKING FOR CONFIG FILES:")
config_files = []
for root, dirs, files in os.walk(tcpformer_root):
    for file in files:
        if (file.endswith(('.yaml', '.yml', '.json', '.cfg')) or 
            'config' in file.lower()):
            config_files.append(os.path.join(root, file))

for file in config_files[:5]:  # Show first 5
    rel_path = os.path.relpath(file, tcpformer_root)
    print(f"  - {rel_path}")

# 4. Look for 3DHP specific configs
dhp_configs = [f for f in config_files if '3dhp' in f.lower()]
if dhp_configs:
    print(f"\n🎯 Found 3DHP configs:")
    for file in dhp_configs:
        print(f"  - {os.path.relpath(file, tcpformer_root)}")
        
        try:
            with open(file, 'r') as f:
                content = f.read()
            print(f"    Content preview:")
            lines = content.split('\n')[:20]  # First 20 lines
            for line in lines:
                if line.strip() and not line.strip().startswith('#'):
                    print(f"      {line}")
        except Exception as e:
            print(f"    ❌ Error reading: {e}")

# 5. Try to import and inspect the actual data loader
print("\n🔬 ATTEMPTING TO IMPORT DATA LOADER:")
try:
    # Look for common data loader patterns
    possible_imports = [
        'data.data_loader_3dhp',
        'lib.data.data_loader_3dhp', 
        'common.data_loader_3dhp',
        'data_loader_3dhp'
    ]
    
    data_loader_class = None
    for import_path in possible_imports:
        try:
            module = __import__(import_path, fromlist=[''])
            if hasattr(module, 'MPI3DHP'):
                data_loader_class = module.MPI3DHP
                print(f"  ✓ Found MPI3DHP class in {import_path}")
                break
            elif hasattr(module, 'Dataset3DHP'):
                data_loader_class = module.Dataset3DHP
                print(f"  ✓ Found Dataset3DHP class in {import_path}")
                break
        except:
            continue
    
    if data_loader_class:
        print(f"  📋 Class: {data_loader_class}")
        
        # Inspect the class
        import inspect
        methods = [method for method in dir(data_loader_class) if not method.startswith('_')]
        print(f"  📝 Methods: {methods}")
        
        # Look at __init__ and __getitem__ if available
        for method_name in ['__init__', '__getitem__', 'preprocess', 'normalize']:
            if hasattr(data_loader_class, method_name):
                method = getattr(data_loader_class, method_name)
                try:
                    source = inspect.getsource(method)
                    print(f"\n  🔍 {method_name} method:")
                    lines = source.split('\n')[:15]  # First 15 lines
                    for line in lines:
                        print(f"    {line}")
                    if len(source.split('\n')) > 15:
                        print("    ... (truncated)")
                except:
                    print(f"    ❌ Could not get source for {method_name}")
    
except Exception as e:
    print(f"  ❌ Import failed: {e}")

# 6. Check if there are any preprocessing or normalization utilities
print("\n🛠️ LOOKING FOR PREPROCESSING UTILITIES:")
util_files = []
for root, dirs, files in os.walk(tcpformer_root):
    for file in files:
        if (file.endswith('.py') and 
            ('util' in file.lower() or 'preprocess' in file.lower() or 'transform' in file.lower())):
            util_files.append(os.path.join(root, file))

for file in util_files[:5]:  # Show first 5
    rel_path = os.path.relpath(file, tcpformer_root)
    print(f"  - {rel_path}")

# 7. Summary and recommendations
print("\n📊 ANALYSIS SUMMARY:")
print("1. Found data loading files - checking for coordinate processing")
print("2. Looking for normalization or transformation functions")
print("3. Checking config files for data format specifications")
print("\n💡 RECOMMENDATIONS:")
print("- Check the actual data loader implementation for coordinate handling")
print("- Look for any normalization constants or coordinate system assumptions")
print("- Compare coordinate ranges between ground truth and your YOLO output")