"""
cull — Rule-based F1 photo culling package.

Modules
-------
exif_reader   Read EXIF metadata and group burst sequences.
detector      Cascade object detection: F1 YOLO → COCO YOLOv8n fallback.
sharpness     Laplacian-variance sharpness scoring inside detection bboxes.
composition   Composition scoring: fill, rule-of-thirds, lead-room.
scorer        Aggregate scores, group-level TopN selection, Rating mapping.
xmp_writer    Write Lightroom-compatible XMP sidecar files.
"""
