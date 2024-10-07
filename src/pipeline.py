import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline, LCMScheduler
from pipelines.models import TextToImageRequest
from torch import Generator
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)
 
def load_pipeline() -> StableDiffusionXLPipeline:
    try
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "models/newdream-sdxl-20", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda:0")
    except:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "models/newdream-sdxl-20", torch_dtype=torch.float16, use_safetensors=True
        ).to("cuda:0")

    
    prompt = "table, beatiful, school, student, girl"
    pipe.fuse_qkv_projections()
    config = CompilationConfig.Default()
    # xformers and Triton are suggested for achieving best performance.
    try:
        import xformers
        config.enable_xformers = True
    except ImportError:
        print('xformers not installed, skip')
    try:
        import triton
        config.enable_triton = True
    except ImportError:
        print('Triton not installed, skip')

    config.enable_cuda_graph = True

    pipe = compile(pipe, config)
    pipe(prompt)
    return pipe

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None
    def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
        if step_index == int(pipe.num_timesteps * 0.50):
            callback_kwargs['prompt_embeds'] = callback_kwargs['prompt_embeds'].chunk(2)[-1]
            callback_kwargs['add_text_embeds'] = callback_kwargs['add_text_embeds'].chunk(2)[-1]
            callback_kwargs['add_time_ids'] = callback_kwargs['add_time_ids'].chunk(2)[-1]
            pipe._guidance_scale = 0.0

        return callback_kwargs

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
        num_inference_steps=10,
        callback_on_step_end=callback_dynamic_cfg,
        callback_on_step_end_tensor_inputs=['prompt_embeds', 'add_text_embeds', 'add_time_ids'],
    ).images[0]
