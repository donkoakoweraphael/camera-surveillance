# sort.py - version simple pour MVP
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class SimpleTracker:
    def __init__(self, max_lost=10, trace_len=50):
        self.next_id = 1
        self.tracks = {}  # id -> [bbox, lost, deque(trace)]
        self.max_lost = max_lost
        self.trace_len = trace_len

    def _centroid(self, b):
        return [int((b[0]+b[2])//2), int((b[1]+b[3])//2)]

    def _iou(self, a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2-x1)*(y2-y1)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        return inter / (area_a + area_b - inter + 1e-9)

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2,conf,class_name]
        det_boxes = [d[:4] for d in detections]
        det_count = len(det_boxes)
        trk_ids = list(self.tracks.keys())
        trk_count = len(trk_ids)

        if trk_count == 0:
            for d in det_boxes:
                self.tracks[self.next_id] = [d, 0, deque([self._centroid(d)], maxlen=self.trace_len)]
                self.next_id += 1
            return

        # build cost matrix (1 - iou)
        cost = np.ones((trk_count, det_count), dtype=float)
        for i, tid in enumerate(trk_ids):
            for j, d in enumerate(det_boxes):
                cost[i, j] = 1 - self._iou(self.tracks[tid][0], d)

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_trks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < 0.7:
                tid = trk_ids[r]
                self.tracks[tid][0] = det_boxes[c]
                self.tracks[tid][1] = 0
                self.tracks[tid][2].append(self._centroid(det_boxes[c]))
                assigned_trks.add(tid)
                assigned_dets.add(c)

        # new detections -> new tracks
        for j in range(det_count):
            if j not in assigned_dets:
                d = det_boxes[j]
                self.tracks[self.next_id] = [d, 0, deque([self._centroid(d)], maxlen=self.trace_len)]
                self.next_id += 1

        # increment lost for unassigned tracks
        for tid in list(self.tracks.keys()):
            if tid not in assigned_trks:
                self.tracks[tid][1] += 1
                if self.tracks[tid][1] > self.max_lost:
                    del self.tracks[tid]

    def get_tracks(self):
        out = []
        for tid, (bbox, lost, trace) in self.tracks.items():
            out.append((tid, bbox, list(trace)))
        return out
