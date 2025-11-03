import os
import json
import re
import cv2
import time
import sys
from tqdm import tqdm
from insightface.app import FaceAnalysis

app = FaceAnalysis(name='antelopev2', allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0)

pattern = re.compile(r"^(\d{2}05)_img\.jpg$")
source_fold = '/workspace/datasetvol/mvhuman_data/relit_images'

def process_objects(args):
    """Process a list of objects and return their data."""
    print(f"Running process objects {args[1]}")
    obj_list, process_idx = args  # unpack
    local_data = {}

    start_time = time.time()
    for i, obj in enumerate(tqdm(
        obj_list,
        desc=f"Proc {process_idx+1}",
        file=sys.stdout,
        ascii=True,
        dynamic_ncols=False,
        disable=False
    )):
        elapsed_time = time.time() - start_time
        if i % 1 == 0:  # Log every 100 items
            eta = (len(obj_list) - i) * (elapsed_time / max(1, i))
            tqdm.write(f"Proc {process_idx+1}: {i}/{len(obj_list)} processed, ETA: {tqdm.format_interval(eta)}")

        obj_path = os.path.join(source_fold, obj, 'images_lr')
        local_data[obj] = {}
        pattern = re.compile(r"^(\d{2}05)_img\.(png|jpg|jpeg|bmp|gif)$", re.IGNORECASE)

        for cam in os.listdir(obj_path):
            cam_path = os.path.join(obj_path, cam)
            local_data[obj][cam] = {}
            image_list = os.listdir(cam_path)
            selected_images = [
                img for img in image_list
                if pattern.match(img) and int(pattern.match(img).group(1)) % 100 == 5 ]
            for file in selected_images:
                if pattern.match(file):
                    img = cv2.imread(os.path.join(cam_path, file))
                    faces = app.get(img)
                    if len(faces) == 0:
                        continue
                    data = faces[0].embedding.tolist()
                    local_data[obj][cam][file] = data
    return local_data

def get_pod_index():
    """Get the pod index from environment variables."""
    # Try different ways to get pod index
    # hostname = os.environ.get('HOSTNAME', '')
    return int(os.environ.get('JOB_COMPLETION_INDEX', 0))

if __name__ == "__main__":
    object_ranges = range(100001, 100501)
    all_objects = list(object_ranges)
    all_objects = [str(obj) for obj in all_objects]
    all_objects.sort()  # ensure stable order

    # Split into 10 roughly equal chunks
    num_processes = int(os.environ.get("JOB_PARALLELISM", 1))
    chunk_size = len(all_objects) // num_processes
    chunks = [all_objects[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes - 1)]
    chunks.append(all_objects[(num_processes - 1) * chunk_size:])

    # Get pod index
    pod_idnex = get_pod_index()
    object_list = chunks[pod_idnex]

    result = process_objects((object_list, pod_idnex))

    # Save each chunk’s results separately
    output_dir = "/workspace/datasetvol/mvhuman_data/arcface_embeddings/relit_images"
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, f"data_part_{pod_idnex}.json")
    with open(out_path, 'w') as f:
        json.dump(result, f)
    print(f"✅ Saved {out_path}")
