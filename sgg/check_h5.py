"""检查 h5 文件结构"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import h5py
    h5_path = "/home/shi/abschluss/dataset/vg150/VG-SGG-with-attri.h5"
    print(f"Opening {h5_path}...")
    f = h5py.File(h5_path, 'r')
    
    print("\n=== Keys in h5 file ===")
    def print_structure(name, obj):
        if hasattr(obj, 'shape'):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{name}: {type(obj)}")
    
    f.visititems(print_structure)
    
    print("\n=== Top-level keys ===")
    for key in f.keys():
        obj = f[key]
        if hasattr(obj, 'shape'):
            print(f"{key}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{key}: {type(obj)}")
    
    f.close()
    print("\n✅ Check completed")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
