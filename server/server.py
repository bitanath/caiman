import asyncio
import tornado
import subprocess
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline
from diffusers import DiffusionPipeline,AutoPipelineForImage2Image
from utils import florence_caption,enhance_prompt,erode,dilate,merge_bboxes,mask_to_boxes,from_base64,to_base64

import random
import numpy as np
import os
import cv2
from textsegmenter import build_sam,SamPredictor
from heatmapper import Net


subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=dtype, revision="refs/pr/1").to(device)
florence_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).to(device).eval()
with torch.no_grad():
    torch.cuda.empty_cache()
florence_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
enhancer_long = pipeline("summarization", model="gokaygokay/Lamini-Prompt-Enchance-Long", device=device)
with torch.no_grad():
    torch.cuda.empty_cache()

segment_anything = build_sam( encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16, encoder_global_attn_indexes=[5, 11, 17, 23], model_type="vit_l", checkpoint="hi_sam_l.pth" )
segment_anything = segment_anything.to('cuda')
predictor = SamPredictor(segment_anything)
with torch.no_grad():
    torch.cuda.empty_cache()

refiner = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
refiner.enable_model_cpu_offload()
refiner.enable_xformers_memory_efficient_attention()

def generate(img,existing_prompt):
    img.thumbnail((1024,1024))
    seed = random.randint(0, 100000000)
    generator = torch.Generator(device=device).manual_seed(seed)
    prompt = florence_caption(image=img,florence_model=florence_model,florence_processor=florence_processor)
    prompt = enhance_prompt(enhancer_long,prompt)
    prompt = existing_prompt+","+prompt if existing_prompt is not None else prompt
    alternative = pipe( prompt=prompt, generator=generator, num_inference_steps=4, width=img.size[0], height=img.size[1], guidance_scale=0.0 )
    alternate = alternative.images[0]
    predictor.set_image(np.asarray(alternate))
    mask, hr_mask, score, hr_score = predictor.predict(multimask_output=False)
    mref = Image.fromarray(np.array(hr_mask.transpose(1,2,0).squeeze()*255).astype("uint8")).convert("RGB")
    bboxes = mask_to_boxes(hr_mask.transpose(1,2,0).squeeze())
    merged = merge_bboxes(bboxes.tolist())
    h = img.size[1] * 0.02
    arr = []
    for box in merged:
        if(box[3] - box[1] > h):
            masker = mref.crop(box)
            imger = alternate.crop(box)
            composited = Image.composite(imger,Image.new("RGBA",imger.size,(0,0,0,0)),masker.convert("L"))
            arr.append([composited,box])
    
    inpainted = Image.fromarray(cv2.inpaint(np.array(alternate), np.array(dilate(mref.convert("L"),3,2)), 3, cv2.INPAINT_NS))
    with torch.no_grad():
        torch.cuda.empty_cache()

    refined = refiner(prompt, image=inpainted, strength=0.10).images[0]
    return alternate,inpainted,refined,arr

def heatmap(img):
    img.thumbnail((1024,1024))
    heatmapper = Net()
    heatmapper = heatmapper.to('cuda')
    overlaid = Image.fromarray(heatmapper(img).astype("uint8"))
    return overlaid



class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Bad Request")
    
    def post(self):
        auth = self.request.headers.get("authorization")
        if(auth is None):
            self.write({"err":"Unauthenticated Request"})
        args = tornado.escape.json_decode(self.request.body)
        imgBase64 = args.get("imgBase64")
        
        img = from_base64(imgBase64)
        alternate,inpainted,refined,arr = generate(img=img)
        alternateB64 = to_base64(alternate)
        backgroundB64 = to_base64(refined)

        self.write({"alternate":alternateB64,"background":backgroundB64})

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

async def main():
    app = make_app()
    app.listen(8000)
    print("Running on 8000")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())