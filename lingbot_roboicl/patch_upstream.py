"""Install the RoboICL integration into the pinned LingBot-VA deployment."""

from pathlib import Path
import sys


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one upstream match for {old!r}")
    return source.replace(old, new)


root = Path(sys.argv[1])
model = root / "wan_va/modules/model.py"
train = root / "wan_va/train.py"
server = root / "wan_va/wan_va_server.py"

source = model.read_text()
flash_import = """try:
    from flash_attn_interface import flash_attn_func
except:
    from flash_attn import flash_attn_func

"""
if flash_import in source:
    source = replace_once(source, flash_import, "")
    source = replace_once(source, """        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func""", """        elif attn_mode == 'flashattn':
            try:
                from flash_attn_interface import flash_attn_func
            except ImportError:
                from flash_attn import flash_attn_func
            self.attn_op = flash_attn_func""")
    model.write_text(source)
if "def _context_embed(" not in source:
    source = replace_once(source, "        device,\n    ):\n        torch._inductor", "        device,\n        context_length=512,\n    ):\n        torch._inductor")
    source = replace_once(source, "expand(-1, 512).flatten()", "expand(-1, context_length).flatten()")
    source = replace_once(source, "    def forward_train(self, input_dict):", """    def _context_embed(self, input_dict):
        if 'demo_latents' not in input_dict:
            return self._input_embed(input_dict['text_emb'], input_type='text')
        demo = input_dict['demo_latents'].to(self.patch_embedding_mlp.weight.dtype)
        tokens = self._input_embed(demo, input_type='latent')
        # Give each sampled time and spatial position a distinct fixed embedding.
        shape = [size // patch for size, patch in zip(demo.shape[-3:], self.patch_size)]
        position = torch.cartesian_prod(*[
            torch.arange(size, device=tokens.device) for size in shape
        ]).float()
        axis_width = tokens.shape[-1] // 6
        frequencies = 10000 ** (-torch.arange(axis_width, device=tokens.device).float() / axis_width)
        phase = position[..., None] * frequencies
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(1)
        embedding = F.pad(embedding, (0, tokens.shape[-1] - embedding.shape[-1]))
        return tokens + embedding.to(tokens.dtype)

    def forward_train(self, input_dict):""")
    source = replace_once(source, 'self._input_embed(latent_dict["text_emb"], input_type=\'text\')', 'self._context_embed(latent_dict)\n        context_length = text_hidden_states.shape[1]')
    source = replace_once(source, "                               device=hidden_states.device\n", "                               device=hidden_states.device,\n                               context_length=context_length,\n")
    source = replace_once(source, 'self.condition_embedder.text_embedder(\n            input_dict["text_emb"])', 'self._context_embed(input_dict)')
    model.write_text(source)

source = train.read_text()
if "RoboICLDataset" not in source:
    source = replace_once(source, "        train_dataset = MultiLatentLeRobotDataset(config=config)", """        if getattr(config, 'roboicl_work', None):
            from roboicl_data import RoboICLDataset

            train_dataset = RoboICLDataset(
                config.roboicl_work, 'train',
                Path(config.wan22_pretrained_model_name_or_path) / 'vae',
                max_frames=config.roboicl_max_frames,
            )
        else:
            train_dataset = MultiLatentLeRobotDataset(config=config)""")
    source = replace_once(source, """        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']""", """        if 'demo_latents' in batch_dict:
            latent_dict['demo_latents'] = batch_dict['demo_latents']
            latent_dict['noisy_latents'][:, :, :1] = batch_dict['latents'][:, :, :1]
            latent_dict['latent'][:, :, :1] = batch_dict['latents'][:, :, :1]
            latent_dict['timesteps'][:, :1] = 0
            latent_dict['cond_timesteps'][:, :1] = 0
        else:
            latent_dict['text_emb'] = batch_dict['text_emb']
            action_dict['text_emb'] = batch_dict['text_emb']""")
    source = replace_once(source, """        return input_dict

    def convert_input_format""", """        if 'demo_latents' in batch_dict:
            input_dict['chunk_size'] = self.config.frame_chunk_size
            input_dict['window_size'] = self.config.attn_window
        return input_dict

    def convert_input_format""")
    source = replace_once(source, """        # Frame-wise video loss calculation
""", """        if 'demo_latents' in input_dict['latent_dict']:
            latent_pred = latent_pred[:, :, 1:]
            latent_loss_weight = latent_loss_weight[:, 1:]
        # Frame-wise video loss calculation
""")
    source = replace_once(source, """        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')""", """        latent_target = input_dict['latent_dict']['targets'][:, :, -latent_pred.shape[2]:]
        latent_loss = F.mse_loss(latent_pred.float(), latent_target.float().detach(), reduction='none')""")
    source = replace_once(source, """                if self.step % self.config.save_interval == 0:""", """                if self.step % self.config.save_interval == 0 or (
                    getattr(self.config, 'roboicl_work', None) and self.step == self.config.num_steps
                ):""")
    train.write_text(source)

source = server.read_text()
if 'np.save(os.path.join(self.save_root, "actions.npy")' not in source:
    line = "        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)\n"
    source = replace_once(source, line, line + '        np.save(os.path.join(self.save_root, "actions.npy"), pred_action.numpy())\n')
    server.write_text(source)
if "transformer_checkpoint" not in source:
    source = replace_once(source, """            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),""", """            getattr(job_config, 'transformer_checkpoint', None)
            or os.path.join(job_config.wan22_pretrained_model_name_or_path, 'transformer'),""")
    server.write_text(source)

source = train.read_text()
if "install_lora" not in source:
    source = replace_once(source, """        self.transformer = load_transformer(
""", """        lora_rank = getattr(config, 'lora_rank', 0)
        self.transformer = load_transformer(
""")
    source = replace_once(source, "            torch_dtype=torch.float32,", """            torch_dtype=self.dtype if lora_rank else torch.float32,""")
    source = replace_once(source, """        logger.info("Setting up activation checkpointing ...")""", """        if lora_rank:
            from lora import install_lora

            install_lora(self.transformer, lora_rank, config.lora_alpha, config.lora_seed)
            count = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
            logger.info(f"RoboICL LoRA trainable parameters: {count:,}")

        logger.info("Setting up activation checkpointing ...")""")
    source = replace_once(source, "        self.transformer.requires_grad_(True)", """        if not lora_rank:
            self.transformer.requires_grad_(True)""")
    source = replace_once(source, """        \"\"\"Save model checkpoint in the same format as pretrained model.\"\"\"
        try:""", """        \"\"\"Save the configured adapter or full transformer checkpoint.\"\"\"
        if getattr(self.config, 'lora_rank', 0):
            from lora import save_lora

            directory = self.save_dir / f"checkpoint_step_{self.step}"
            save_lora(self.transformer, directory, self.config.lora_rank,
                      self.config.lora_alpha, self.config.rank == 0)
            return
        try:""")
    train.write_text(source)

source = server.read_text()
if "merge_lora" not in source:
    source = replace_once(source, """        self.transformer = load_transformer(""", """        lora_path = getattr(job_config, 'lora_checkpoint', None)
        self.transformer = load_transformer(""")
    source = replace_once(source, """            torch_device=self.device,
            attn_mode="torch"
        )
        shard_fn""", """            torch_device='cpu' if lora_path else self.device,
            attn_mode="torch"
        )
        if lora_path:
            from lora import merge_lora

            merge_lora(self.transformer, lora_path)
            self.transformer.to(self.device)
        shard_fn""")
    server.write_text(source)

source = train.read_text()
if "EpochSampler" not in source:
    source = replace_once(source, "            count = sum(p.numel()", """            if getattr(config, 'restart_directory', None):
                from lora import ADAPTER_FILE, load_lora

                load_lora(self.transformer, Path(config.restart_directory) / ADAPTER_FILE,
                          lora_rank, config.lora_alpha)
            count = sum(p.numel()""")
    source = replace_once(source, "        self.train_loader = DataLoader(", """        loader_generator = None
        if getattr(config, 'epoch_updates', 0):
            from epoch_state import EpochSampler

            train_sampler = EpochSampler(len(train_dataset), config.world_size, config.rank,
                                         config.gradient_accumulation_steps, config.lora_seed)
            loader_generator = torch.Generator().manual_seed(config.lora_seed + config.rank)
        self.train_loader = DataLoader(""")
    source = replace_once(source, "            sampler=train_sampler,", """            sampler=train_sampler,
            generator=loader_generator,""")
    source = replace_once(source, "        self.train_loader_iter = None\n", """        self.train_loader_iter = None
        if getattr(config, 'restart_directory', None):
            from epoch_state import restore_restart

            restore_restart(self, Path(config.restart_directory))
""")
    source = replace_once(source, "        if getattr(self.config, 'lora_rank', 0):\n", """        if getattr(self.config, 'epoch_updates', 0):
            from epoch_state import save_restart

            save_restart(self)
            return
        if getattr(self.config, 'lora_rank', 0):
""")
    source = replace_once(source, "                    getattr(self.config, 'roboicl_work', None) and self.step == self.config.num_steps", """                    getattr(self.config, 'epoch_updates', 0)
                    and self.step % self.config.epoch_updates == 0
                ) or (
                    getattr(self.config, 'roboicl_work', None) and self.step == self.config.num_steps""")
    train.write_text(source)

source = train.read_text()
if "first RoboICL update" not in source:
    source = replace_once(source, "lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps)", """lr_lambda=lambda step: warmup_constant_lambda(
                step + int(bool(getattr(config, 'epoch_updates', 0))), warmup_steps=config.warmup_steps
            )""")
    source = replace_once(source, "            self.optimizer.step()", """            if getattr(self.config, 'epoch_updates', 0) and self.step == 0:
                if not torch.isfinite(total_norm).item():
                    raise RuntimeError("Non-finite gradient at the first RoboICL update")
            self.optimizer.step()""")
    source = replace_once(source, "                    and self.step % self.config.epoch_updates == 0", """                    and (self.step == 1 or self.step % self.config.epoch_updates == 0)""")
    train.write_text(source)

source = train.read_text()
if "reserve_training_memory" not in source:
    source = replace_once(source, """        while self.step < self.config.num_steps:""", """        reserve_gib = int(os.environ.get('ROBOICL_RESERVE_GIB', '0')) if getattr(self.config, 'epoch_updates', 0) else 0
        if reserve_gib and self.step < self.config.num_steps:
            from epoch_state import reserve_training_memory

            reserve_training_memory(self.device, reserve_gib)

        while self.step < self.config.num_steps:""")
    source = replace_once(source, """                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()""", """                if reserve_gib:
                    reserve_training_memory(self.device, reserve_gib)
                elif self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()""")
    train.write_text(source)
