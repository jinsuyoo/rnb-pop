import os
import time
from multiprocessing import Process, Queue

import numpy as np

from opencood.visualization import simple_vis


class DataSplitSaver:
    def __init__(self):
        self.n_processes = 12

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    #model_one_pred, model_two_pred, gt, origin_lidar, gt_range, vis_save_path, distance = queue.get()
                    saver_dict = queue.get()

                    #if vis_save_path is None:
                    #    break
                    simple_vis.visualize_data_split(
                        saver_dict['model_one_pred'], 
                        saver_dict['model_two_pred'], 
                        saver_dict['gt_bbox'], 
                        saver_dict['ref_gt_bbox'],
                        saver_dict['refcar_bbox'],
                        saver_dict['ptc_np'], 
                        saver_dict['canvas_range'], 
                        saver_dict['vis_save_path'],
                        method='bev', left_hand=False, vis_pred_box=True, vis_ref_car=True,
                        canvas_bg_color=(255, 255, 255), distance=saver_dict['distance']
                    )

        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]

        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(self, save_dict):
        self.queue.put((save_dict))


class SaverV2:
    def __init__(self):
        self.n_processes = 8

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    pred_box_t, gt_box_t, origin_lidar, gt_range, vis_save_path = queue.get()
                    if vis_save_path is None:
                        break
                    simple_vis.visualize(
                        pred_box_t, gt_box_t, origin_lidar, 
                        gt_range, vis_save_path,
                        method='bev', left_hand=False, vis_pred_box=True, vis_ref_car=True,
                        canvas_bg_color=(255, 255, 255)
                    )

        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]

        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(
        self, model_dir, i, pred_box_t, gt_box_t, origin_lidar, gt_range, folder_name, timestep):
        
        vis_save_path = os.path.join(model_dir, 'vis_bev', folder_name)
        os.makedirs(vis_save_path, exist_ok=True)
        vis_save_path = os.path.join(vis_save_path, f'{timestep:06d}.png')
        self.queue.put((pred_box_t, gt_box_t, origin_lidar, gt_range, vis_save_path))


class Saver:
    def __init__(self):
        self.n_processes = 8

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    pred_box_t, gt_box_t, refined_box_t, discovered_ref_pred_t, refcar_bbox_t, origin_lidar, gt_range, vis_save_path = queue.get()
                    if vis_save_path is None:
                        break
                    simple_vis.visualize(
                        pred_box_t, gt_box_t, origin_lidar, 
                        gt_range, vis_save_path,
                        refined_ref_pred_np=refined_box_t,
                        discovered_ref_pred_np=discovered_ref_pred_t,
                        refcar_np=refcar_bbox_t,
                        method='bev', left_hand=False, vis_pred_box=True, vis_ref_car=True,
                        canvas_bg_color=(255, 255, 255)
                    )

        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]

        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(
        self, 
        model_dir, 
        i, 
        pred_box_t, 
        gt_box_t, 
        refined_box_t,
        discovered_ref_pred_t,
        refcar_bbox_t,
        origin_lidar, gt_range, folder_name, timestep):
        
        vis_save_path = os.path.join(model_dir, 'vis_bev', folder_name)
        os.makedirs(vis_save_path, exist_ok=True)
        vis_save_path = os.path.join(vis_save_path, f'{timestep:06d}.png')
        self.queue.put((
            pred_box_t, gt_box_t, refined_box_t,
            discovered_ref_pred_t,
            refcar_bbox_t,
            origin_lidar, gt_range, vis_save_path))
