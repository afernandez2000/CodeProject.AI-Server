from insightface.utils import face_align

def align_face(bgr_img, kps, image_size=112):
    return face_align.norm_crop(bgr_img, kps, image_size=image_size)
