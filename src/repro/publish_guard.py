"""Guard for publishing images that show UTA-RLDD faces.

Only the subjects listed in configs/image_publishable.csv (the 'YES' rows of the official
UTA-RLDD page) may have face images made public. Any code that saves an image showing a
face must go through :func:`save_face_image`, which refuses other subjects.
"""
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUBLISHABLE_CSV = os.path.join(REPO, "configs", "image_publishable.csv")

_cache = None


class NotPublishableError(PermissionError):
    """Raised when a face image of a subject who did not consent to publication would be saved."""


def _publishable_ids():
    global _cache
    if _cache is None:
        with open(PUBLISHABLE_CSV, "r") as f:
            _cache = frozenset(int(row["subject_id"]) for row in csv.DictReader(f))
    return _cache


def is_publishable(subject_id):
    """True if face images of this UTA-RLDD subject (int or numeric str) may be published."""
    try:
        return int(subject_id) in _publishable_ids()
    except (TypeError, ValueError):
        return False


def save_face_image(image, path, subject_id, writer=None):
    """Save an image showing a face, but only for publishable subjects.

    ``writer(path, image)`` defaults to ``cv2.imwrite``. Raises NotPublishableError (nothing is
    written) if ``subject_id`` is not in configs/image_publishable.csv or is missing/invalid.
    """
    if not is_publishable(subject_id):
        raise NotPublishableError("subject_id=%r is not listed in %s; refusing to save a face image"
                                  % (subject_id, PUBLISHABLE_CSV))
    if writer is None:
        import cv2
        writer = cv2.imwrite
    return writer(path, image)
