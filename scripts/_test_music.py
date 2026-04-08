import importlib.util
import psutil

spec = importlib.util.spec_from_file_location('mc', 'brain/tools/generated/music_controller_v1.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("=== now_playing (Apple Music launched) ===")
r = mod.music_controller("now_playing")
print(r)

print("\n=== Detected media processes ===")
for p in psutil.process_iter(["pid", "name"]):
    n = p.info["name"] or ""
    if "apple" in n.lower() or "music" in n.lower() or "itunes" in n.lower():
        print(f"  {n} (PID {p.info['pid']})")
