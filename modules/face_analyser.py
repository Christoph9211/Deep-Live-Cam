import os
import shutil
from typing import Any
import insightface

import cv2
import numpy as np
import modules.globals
from tqdm import tqdm
from modules.typing import Frame
from modules.cluster_analysis import find_cluster_centroids, find_closest_centroid
from modules.utilities import get_temp_directory_path, create_temp, extract_frames, clean_temp, get_temp_frame_paths
from pathlib import Path

FACE_ANALYSER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers)
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None

def has_valid_map() -> bool:
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            return True
    return False

def default_source_face() -> Any:
    for map in modules.globals.source_target_map:
        if "source" in map:
            return map['source']['face']
    return None

def simplify_maps() -> Any:
    centroids = []
    faces = []
    for map in modules.globals.source_target_map:
        if "source" in map and "target" in map:
            centroids.append(map['target']['face'].normed_embedding)
            faces.append(map['source']['face'])

    modules.globals.simple_map = {'source_faces': faces, 'target_embeddings': centroids}
    return None

def add_blank_map() -> Any:
    try:
        max_id = -1
        if len(modules.globals.source_target_map) > 0:
            max_id = max(modules.globals.source_target_map, key=lambda x: x['id'])['id']

        modules.globals.source_target_map.append({
                'id' : max_id + 1
                })
    except ValueError:
        return None
    
def get_unique_faces_from_target_image() -> Any:
    try:
        modules.globals.source_target_map = []
        target_frame = cv2.imread(modules.globals.target_path)
        many_faces = get_many_faces(target_frame)
        i = 0

        for face in many_faces:
            x_min, y_min, x_max, y_max = face['bbox']
            modules.globals.source_target_map.append({
                'id' : i, 
                'target' : {
                            'cv2' : target_frame[int(y_min):int(y_max), int(x_min):int(x_max)],
                            'face' : face
                            }
                })
            i = i + 1
    except ValueError:
        return None
    
    
def get_unique_faces_from_target_video() -> Any:
    """Extract unique faces from the target video without keeping all frames in memory."""
    try:
        modules.globals.source_target_map = []
        frame_face_embeddings = []
        face_embeddings = []

        # Extract frames to a temporary directory
        create_temp(modules.globals.target_path)
        extract_frames(modules.globals.target_path)
        frame_paths = sorted(get_temp_frame_paths(modules.globals.target_path))

        progress = tqdm(frame_paths, desc="Extracting face embeddings from frames")
        for i, frame_path in enumerate(progress):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            many_faces = get_many_faces(frame)

            for face in many_faces:
                face_embeddings.append(face.normed_embedding)

            frame_face_embeddings.append({'frame': i, 'faces': many_faces, 'location': frame_path})

        progress.close()

        centroids = find_cluster_centroids(face_embeddings)
        del face_embeddings  # free memory

        for frame in frame_face_embeddings:
            for face in frame['faces']:
                closest_centroid_index, _ = find_closest_centroid(centroids, face.normed_embedding)
                face['target_centroid'] = closest_centroid_index

        frame_paths_map = {idx: path for idx, path in enumerate(frame_paths)}

        for idx in range(len(centroids)):
            modules.globals.source_target_map.append({'id': idx})

            temp = []
            for frame in tqdm(frame_face_embeddings, desc=f"Mapping frame embeddings to centroids-{idx}"):
                filtered = [f for f in frame['faces'] if f['target_centroid'] == idx]
                if filtered:
                    temp.append({'frame': frame['frame'], 'location': frame['location'], 'faces': filtered})

            modules.globals.source_target_map[idx]['target_faces_in_frame'] = temp

        default_target_face(frame_paths_map)
        clean_temp(modules.globals.target_path)
    except ValueError:
        return None
    

def default_target_face(frame_images: dict) -> None:
    """Select a representative target face for each mapping."""
    for mapping in modules.globals.source_target_map:
        best_face = None
        best_frame_idx = None
        for frame in mapping['target_faces_in_frame']:
            if len(frame['faces']) > 0:
                best_face = frame['faces'][0]
                best_frame_idx = frame['frame']
                break

        for frame in mapping['target_faces_in_frame']:
            for face in frame['faces']:
                if face['det_score'] > best_face['det_score']:
                    best_face = face
                    best_frame_idx = frame['frame']

        x_min, y_min, x_max, y_max = best_face['bbox']

        target_frame_path = frame_images.get(best_frame_idx)
        target_cv2 = cv2.imread(target_frame_path)
        if target_cv2 is None:
            continue
        mapping['target'] = {
            'cv2': target_cv2[int(y_min):int(y_max), int(x_min):int(x_max)],
            'face': best_face,
        }


def dump_faces(centroids: Any, frame_face_embeddings: list):
    temp_directory_path = get_temp_directory_path(modules.globals.target_path)

    for i in range(len(centroids)):
        if os.path.exists(temp_directory_path + f"/{i}") and os.path.isdir(temp_directory_path + f"/{i}"):
            shutil.rmtree(temp_directory_path + f"/{i}")
        Path(temp_directory_path + f"/{i}").mkdir(parents=True, exist_ok=True)

        for frame in tqdm(frame_face_embeddings, desc=f"Copying faces to temp/./{i}"):
            temp_frame = cv2.imread(frame['location'])

            j = 0
            for face in frame['faces']:
                if face['target_centroid'] == i:
                    x_min, y_min, x_max, y_max = face['bbox']

                    if temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)].size > 0:
                        cv2.imwrite(temp_directory_path + f"/{i}/{frame['frame']}_{j}.png", temp_frame[int(y_min):int(y_max), int(x_min):int(x_max)])
                j += 1
