import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import modules
import modules.globals
from queue import Queue, Empty
FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_image',
    'process_video',
    'process_frame_stream'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'modules.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                sys.exit()
    except ImportError:
        print(f"Frame processor {frame_processor} not found")
        sys.exit()
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES

def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    """Add or remove frame processor modules from FRAME_PROCESSORS_MODULES based on the ui switch state.

    Args:
        frame_processors (List[str]): List of frame processor names
    """
    global FRAME_PROCESSORS_MODULES
    for frame_processor, state in modules.globals.fp_ui.items():
        if state == True and frame_processor not in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
            modules.globals.frame_processors.append(frame_processor)
        if state == False:
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.remove(frame_processor_module)
                modules.globals.frame_processors.remove(frame_processor)
            except:
                pass

def multi_process_frame(
    source_path: str,
    temp_frame_paths: List[str],
    process_frames: Callable[[str, List[str], Any], None],
    progress: Any = None,
    batch_size: int = 64,
) -> None:

    # Ensure batch_size and max_workers are valid
    """
    Process frames in parallel using a thread pool.

    Args:
        source_path (str): Path to the source image or video.
        temp_frame_paths (List[str]): Paths to the temporary frames of the video.
        process_frames (Callable[[str, List[str], Any], None]): Function to process a batch of frames.
        progress (Any, optional): Optional progress object to track the progress of the processing.
        batch_size (int, optional): Number of frames to process in each batch. Defaults to 64.

    Returns:
        None
    """

    effective_batch_size = max(1, batch_size)
    max_workers = max(1, modules.globals.execution_threads or 1)

    frame_queue: Queue[str] = Queue()
    for path in temp_frame_paths:
        frame_queue.put(path)

    def worker() -> None:
        """
        Worker function for multi_process_frame.

        This function is a target for threads in a ThreadPoolExecutor. It will
        continually pull frames from the frame_queue and process them in batches
        until the queue is empty.

        Args:
            None

        Returns:
            None
        """
        while True:
            batch: List[str] = []
            # Pull up to ``effective_batch_size`` items without blocking.  We
            # cannot simply break out of the loop on ``Empty`` because any
            # frames already dequeued would be dropped and never processed.
            for _ in range(effective_batch_size):
                try:
                    batch.append(frame_queue.get_nowait())
                except Empty:
                    break

            if not batch:
                # Queue is empty and nothing was dequeued for this batch, so
                # the worker can exit cleanly.
                break

            process_frames(source_path, batch, progress)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker) for _ in range(max_workers)]
        for future in futures:
            future.result() # Wait for all workers to complete

# def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], progress: Any = None) -> None:
#     with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
#         futures = []
#         for path in temp_frame_paths:
#             future = executor.submit(process_frames, source_path, [path], progress)
#             futures.append(future)
#         for future in futures:
#             future.result()


def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None], batch_size: int = 64) -> None:
    """
    Process frames in batches using a configurable number of worker threads.
    This approach is efficient by reducing task creation overhead.

    Args:
        source_path (str): Source path of the video/image
        frame_paths (list[str]): List of frame paths to process
        process_frames (Callable[[str, List[str], Any], None]): Function to process frames
        batch_size (int, optional): Number of frames to process in a single batch. Defaults to 64.
    """

    progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    total = len(frame_paths)
    with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
        # Build a compact flags string without depending on modules.core to avoid circular imports
        g = modules.globals
        flags = []
        if getattr(g, 'mouth_mask', False):
            flags.append('MM')
        if getattr(g, 'preserve_teeth', False):
            flags.append('TTH')
        if getattr(g, 'preserve_hairline', False):
            flags.append('HL')
        if g.fp_ui.get('face_enhancer', False):
            flags.append('ENH')
        if getattr(g, 'map_faces', False):
            flags.append('MAP')
        if getattr(g, 'many_faces', False):
            flags.append('MF')
        if getattr(g, 'smoothing_enabled', False):
            flags.append('S')
        flags_str = ','.join(flags) if flags else 'none'
        progress.set_postfix({
            'providers': g.execution_providers,
            'threads': g.execution_threads,
            'mem_gb': g.max_memory,
            'backend': getattr(g, 'segmenter_backend', 'auto'),
            'flags': flags_str,
        })
        multi_process_frame(source_path, frame_paths, process_frames, progress, batch_size)
