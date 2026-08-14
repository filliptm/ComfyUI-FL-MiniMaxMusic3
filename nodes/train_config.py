class FL_MiniMaxMusic3TrainConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_rank": (["16", "32", "64", "128", "256"], {"default": "64"}),
                "lora_alpha": ("INT", {"default": 64, "min": 1, "max": 512}),
                "lora_dropout": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.01}),
                "learning_rate": ("FLOAT", {"default": 0.00005, "min": 0.00000001, "max": 0.1, "step": 0.000001, "round": 0.00000001}),
                "max_train_steps": ("INT", {"default": 500, "min": 1, "max": 1000000}),
                "train_batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "gradient_accumulation_steps": ("INT", {"default": 4, "min": 1, "max": 1024}),
                "lr_scheduler": (["cosine", "constant", "constant_with_warmup", "linear", "polynomial"], {"default": "cosine"}),
                "lr_warmup_steps": ("INT", {"default": 50, "min": 0, "max": 100000}),
                "weight_decay": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "optimizer": (["adamw_bf16", "optimi-stableadamw", "bnb-adamw8bit"], {"default": "adamw_bf16"}),
                "base_model_precision": (["int8-quanto", "no_change"], {"default": "int8-quanto"}),
                "text_encoder_precision": (["int8-quanto", "no_change"], {"default": "int8-quanto"}),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "checkpoint_interval": ("INT", {"default": 100, "min": 1, "max": 100000}),
                "checkpoints_total_limit": ("INT", {"default": 3, "min": 1, "max": 100}),
                "preserve_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FL_MINIMAX_MUSIC3_TRAIN_CONFIG",)
    RETURN_NAMES = ("train_config",)
    FUNCTION = "build"
    CATEGORY = "FL/MiniMax Music 3/Training"
    DESCRIPTION = "Builds a versioned MiniMax Music 3 LoRA configuration for the managed SimpleTuner backend."

    def build(self, lora_rank, lora_alpha, lora_dropout, learning_rate, max_train_steps, train_batch_size, gradient_accumulation_steps, lr_scheduler, lr_warmup_steps, weight_decay, optimizer, base_model_precision, text_encoder_precision, gradient_checkpointing, seed, checkpoint_interval, checkpoints_total_limit, preserve_cache):
        rank = int(lora_rank)
        if lora_alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        return ({
            "schema_version": 1,
            "lora_rank": rank,
            "lora_alpha": int(lora_alpha),
            "lora_dropout": float(lora_dropout),
            "learning_rate": float(learning_rate),
            "max_train_steps": int(max_train_steps),
            "train_batch_size": int(train_batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "effective_batch_size": int(train_batch_size) * int(gradient_accumulation_steps),
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": int(lr_warmup_steps),
            "weight_decay": float(weight_decay),
            "optimizer": optimizer,
            "mixed_precision": "bf16",
            "base_model_precision": base_model_precision,
            "text_encoder_1_precision": text_encoder_precision,
            "gradient_checkpointing": bool(gradient_checkpointing),
            "seed": int(seed),
            "checkpoint_step_interval": int(checkpoint_interval),
            "checkpoints_total_limit": int(checkpoints_total_limit),
            "preserve_data_backend_cache": bool(preserve_cache),
        },)


class FL_MiniMaxMusic3ValidationConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {"default": "funky house groove, 125 bpm, warm bassline, crisp drums", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "duration": ("INT", {"default": 15, "min": 5, "max": 60}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "guidance": ("FLOAT", {"default": 1.7, "min": 0.0, "max": 20.0, "step": 0.1}),
                "validation_interval": ("INT", {"default": 100, "min": 1, "max": 100000}),
                "samples": ("INT", {"default": 1, "min": 1, "max": 8}),
            }
        }

    RETURN_TYPES = ("FL_MINIMAX_MUSIC3_VALIDATION_CONFIG",)
    RETURN_NAMES = ("validation_config",)
    FUNCTION = "build"
    CATEGORY = "FL/MiniMax Music 3/Training"
    DESCRIPTION = "Defines deterministic validation music generated while a Music 3 LoRA trains."

    def build(self, caption, lyrics, duration, seed, inference_steps, guidance, validation_interval, samples):
        caption = caption.strip()
        if not caption:
            raise ValueError("Validation caption cannot be empty")
        return ({
            "schema_version": 1,
            "caption": caption,
            "lyrics": lyrics.strip(),
            "duration": int(duration),
            "seed": int(seed),
            "inference_steps": int(inference_steps),
            "guidance": float(guidance),
            "interval": int(validation_interval),
            "samples": int(samples),
        },)
