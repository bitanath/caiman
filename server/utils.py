import copy
import numpy as np
import cv2

from PIL import Image
from io import BytesIO
import base64

def to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def to_base64_png(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def from_base64(base64_str):
    decoded = base64.b64decode(base64_str)
    im = Image.open(BytesIO(decoded))
    return im


def florence_caption(florence_processor,florence_model,image,device):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    inputs = florence_processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to(device)
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=78,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = florence_processor.post_process_generation(
        generated_text,
        task="<MORE_DETAILED_CAPTION>",
        image_size=(image.width, image.height)
    )
    return parsed_answer["<MORE_DETAILED_CAPTION>"]

# Prompt Enhancer function
def enhance_prompt(enhancer_long,input_prompt):
    result = enhancer_long("Enhance the description: " + input_prompt)
    enhanced_text = result[0]['summary_text']
    return enhanced_text

def erode(image,radius=5,iterations=2):
    img = np.array(image)
    kernel = np.ones((radius,radius),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = iterations)
    return Image.fromarray(erosion)

def dilate(image,radius=5,iterations=2):
    img = np.array(image)
    kernel = np.ones((radius,radius),np.uint8)
    erosion = cv2.dilate(img,kernel,iterations = iterations)
    return Image.fromarray(erosion)

def merge_bboxes(bboxes, delta_x=0.1, delta_y=0.1):


    def is_in_bbox(point, bbox):
        return point[0] >= bbox[0] and point[0] <= bbox[2] and point[1] >= bbox[1] and point[1] <= bbox[3]

    def intersect(bbox, bbox_):
        for i in range(int(len(bbox) / 2)):
            for j in range(int(len(bbox) / 2)):
                # Check if one of the corner of bbox inside bbox_
                if is_in_bbox([bbox[2 * i], bbox[2 * j + 1]], bbox_):
                    return True
        return False
    
    def good_intersect(bbox, bbox_):
        return (bbox[0] < bbox_[2] and bbox[2] > bbox_[0] and
                bbox[1] < bbox_[3] and bbox[3] > bbox_[1])

    # Sort bboxes by ymin
    bboxes = sorted(bboxes, key=lambda x: x[1])

    tmp_bbox = None
    while True:
        nb_merge = 0
        used = []
        new_bboxes = []
        # Loop over bboxes
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):
                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue
                # Compute the bboxes with a margin
                bmargin = [
                    b[0] - (b[2] - b[0]) * delta_x, b[1] - (b[3] - b[1]) * delta_y,
                    b[2] + (b[2] - b[0]) * delta_x, b[3] + (b[3] - b[1]) * delta_y
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * delta_x, b_[1] - (b[3] - b[1]) * delta_y,
                    b_[2] + (b_[2] - b_[0]) * delta_x, b_[3] + (b_[3] - b_[1]) * delta_y
                ]
                
                if intersect(bmargin, b_margin) or intersect(b_margin, bmargin):
                    tmp_bbox = [min(b[0], b_[0]), min(b[1], b_[1]), max(b_[2], b[2]), max(b[3], b_[3])]
                    used.append(j)
                    nb_merge += 1
                if tmp_bbox:
                    b = tmp_bbox
            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)
            used.append(i)
            tmp_bbox = None
        
        if nb_merge == 0:
            break
        bboxes = copy.deepcopy(new_bboxes)

    return new_bboxes

def mask_to_boxes(mask) :
    max_ix = max(s+1 for s in mask.shape)
    x_ixs = np.full(mask.shape, fill_value=max_ix)
    y_ixs = np.full(mask.shape, fill_value=max_ix)

    
    for i in range(mask.shape[0]):
        x_fill_ix = max_ix
        for j in range(mask.shape[1]):
            above_cell_ix = x_ixs[i-1, j] if i>0 else max_ix
            still_active = mask[i, j] or ((x_fill_ix != max_ix) and (above_cell_ix != max_ix))
            x_fill_ix = min(x_fill_ix, j, above_cell_ix) if still_active else max_ix
            x_ixs[i, j] = x_fill_ix

    for j in range(mask.shape[1]):
        y_fill_ix = max_ix
        for i in range(mask.shape[0]):
            left_cell_ix = y_ixs[i, j-1] if j>0 else max_ix
            still_active = mask[i, j] or ((y_fill_ix != max_ix) and (left_cell_ix != max_ix))
            y_fill_ix = min(y_fill_ix, i, left_cell_ix) if still_active else max_ix
            y_ixs[i, j] = y_fill_ix

    
    new_xstops = np.diff((x_ixs != max_ix).astype(np.int32), axis=1, append=False)==-1
    new_ystops = np.diff((y_ixs != max_ix).astype(np.int32), axis=0, append=False)==-1
    corner_mask = new_xstops & new_ystops
    y_stops, x_stops = np.array(np.nonzero(corner_mask))

    x_starts = x_ixs[y_stops, x_stops]
    y_starts = y_ixs[y_stops, x_stops]
    ltrb_boxes = np.hstack([x_starts[:, None], y_starts[:, None], x_stops[:, None]+1, y_stops[:, None]+1])
    return ltrb_boxes