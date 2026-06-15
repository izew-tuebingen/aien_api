import modal
import os

app = modal.App()
volume = modal.Volume.from_name("aien-vector-store", create_if_missing=True)

@app.function(volumes={"/data": volume})
def test():
    print("Testing write access to /data")
    try:
        os.makedirs("/data/test_dir", exist_ok=True)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")
        
    print(f"Directory permissions for /data: {oct(os.stat('/data').st_mode)}")

if __name__ == "__main__":
    test.remote()
