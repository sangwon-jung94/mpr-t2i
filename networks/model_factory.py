import torch



class ModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_model(modelname, train=True):
        cache_dir='/n/holylabs/LABS/calmon_lab/Lab/diffusion_models'
        if modelname == 'BLIP':
            from transformers import BlipForQuestionAnswering
            network = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        elif modelname == 'CLIP':
            import clip
            network, _ = clip.load("ViT-B/32", device= 'cpu')
        elif modelname == 'SD_14':
            from diffusers import StableDiffusionPipeline
            if train:
                network = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", cache_dir=cache_dir)#, torch_dtype=torch.float16)
            else:
                network = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, cache_dir=cache_dir)
        elif modelname == 'SD_15':
            from diffusers import StableDiffusionPipeline
            if train:
                network = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", cache_dir=cache_dir)#, torch_dtype=torch.float16)
            else:
                network = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, cache_dir=cache_dir)
        elif modelname == 'SD_2':
            from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
            model_id ="stabilityai/stable-diffusion-2-1-base"
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            network = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16, cache_dir=cache_dir)
        elif modelname == 'SDXL':
            from diffusers import AutoPipelineForText2Image
            network = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, 
                                                                variant="fp16", use_safetensors=True, cache_dir=cache_dir)
        elif modelname == 'LCM':
            from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
            unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir)
            pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            network = pipe
        elif modelname == 'cascade':
            from diffusers import StableCascadeCombinedPipeline,StableCascadePriorPipeline,StableCascadeDecoderPipeline
            # pipe = StableCascadeCombinedPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.bfloat16)
            prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
            decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)
            prior = prior.to('cuda')
            decoder = decoder.to('cuda')
            network = (prior, decoder)
            # pipe = pipe.to(torch.float16)
            # network = pipe
        elif modelname == 'pixelart':
            from diffusers import PixArtSigmaPipeline
            pipe = PixArtSigmaPipeline.from_pretrained(
                "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                torch_dtype=torch.float16,
                # variant="fp16"
            )
            network = pipe
        elif modelname == 'playground':
            from diffusers import  DiffusionPipeline
            network = DiffusionPipeline.from_pretrained(
                "playgroundai/playground-v2.5-1024px-aesthetic",
                torch_dtype=torch.float16,
                variant="fp16",
            )
        else:
            raise NotImplementedError

        return network
