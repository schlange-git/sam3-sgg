"""检查 h5 文件结构"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import h5py
    # 使用相对路径，从当前脚本位置查找数据集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # 回到项目根目录
    # 尝试多个可能的数据集路径
    possible_paths = [
        os.path.join(project_root, "..", "..", "dataset", "vg150", "VG-SGG-with-attri.h5"),
        os.path.join(project_root, "..", "..", "..", "dataset", "vg150", "VG-SGG-with-attri.h5"),
        os.path.join(os.path.expanduser("~"), "桌面", "abschluss", "sgg", "dataset", "vg150", "VG-SGG-with-attri.h5"),
    ]
    h5_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            h5_path = abs_path
            break
    if h5_path is None:
        # 如果都找不到，使用第一个路径（用户需要自己修改）
        h5_path = os.path.abspath(possible_paths[0])
        print(f"⚠️ 警告: 使用默认路径 {h5_path}，如果不存在请修改脚本")
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
