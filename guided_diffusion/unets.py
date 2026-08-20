from .unet_partial import *


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        ch *= 3
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        model_channels *= 3
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = 288

        self.input_blocks_2 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_3 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_4 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_5 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_6 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_T1 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])

        self.SE_Attention_com = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_2 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_3 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_4 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_5 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_6 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_T1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)

        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 4 * conv_ch, 3 * conv_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch * 3, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, low_res, other):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h1 = x[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        h2 = x[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        h3 = x[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        h4 = low_res[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        h5 = low_res[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        h6 = low_res[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        # h1 = x.type(self.dtype)
        # h2 = low_res.type(self.dtype)

        # print(th.count_nonzero(other))
        # has_other = (th.count_nonzero(other) > 0)
        ht = other.type(self.dtype)
        # print(ht.shape)

        for idx in range(len(self.input_blocks)):
            h1 = self.input_blocks[idx](h1, emb)
            h2 = self.input_blocks_2[idx](h2, emb)
            h3 = self.input_blocks_3[idx](h3, emb)
            h4 = self.input_blocks_4[idx](h4, emb)
            h5 = self.input_blocks_5[idx](h5, emb)
            h6 = self.input_blocks_6[idx](h6, emb)
            ht = self.input_blocks_T1[idx](ht, emb)
            hs.append((1 / 7) * h1 + (1 / 7) * h2 + (1 / 7) * h3 + (1 / 7) * h4 + (1 / 7) * h5 + (1 / 7) * h6 + (1 / 7) * ht)

        com_h1 = self.conv_common(h1)
        com_h2 = self.conv_common(h2)
        com_h3 = self.conv_common(h3)
        com_h4 = self.conv_common(h4)
        com_h5 = self.conv_common(h5)
        com_h6 = self.conv_common(h6)

        dist_h1 = self.conv_distinct(h1)
        dist_h2 = self.conv_distinct(h2)
        dist_h3 = self.conv_distinct(h3)
        dist_h4 = self.conv_distinct(h4)
        dist_h5 = self.conv_distinct(h5)
        dist_h6 = self.conv_distinct(h6)

        dist_h1 = self.SE_Attention_dist_1(dist_h1)
        dist_h2 = self.SE_Attention_dist_2(dist_h2)
        dist_h3 = self.SE_Attention_dist_3(dist_h3)
        dist_h4 = self.SE_Attention_dist_4(dist_h4)
        dist_h5 = self.SE_Attention_dist_5(dist_h5)
        dist_h6 = self.SE_Attention_dist_6(dist_h6)


        com_ht = self.conv_common(ht)
        dist_ht = self.conv_distinct(ht)
        com_h = self.SE_Attention_com((1 / 7) * com_h1 + (1 / 7) * com_h2 + (1 / 7) * com_h3 + (1 / 7) * com_h4 + (1 / 7) * com_h5 + (1 / 7) * com_h6 + (1 / 7) * com_ht)
        dist_ht = self.SE_Attention_dist_3(dist_ht)
        h = th.cat([com_h, dist_h1, dist_h2, dist_h3, dist_h4, dist_h5, dist_h6, dist_ht], dim=1)
        h = self.dim_reduction_non_zeros(h)

        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        return com_h1, com_h2, com_h3, com_h4, com_h5, com_h6, com_ht, dist_h1, dist_h2, dist_h3, dist_h4, dist_h5, dist_h6, dist_ht, self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels, *args, **kwargs)
        print(image_size)
    def forward(self, x, timesteps,**kwargs):
        # _, _, new_height, new_width = x.shape
        # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(x, timesteps, kwargs['low_res'], kwargs['other'])


class UNet2(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels*2, ch, 3, padding=1))]
        )
        # T1 Encoder
        self.input_blocks_T1 = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        ) 

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_T1.append(TimestepEmbedSequential(*layers))# T1 Encoder
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                # T1 Encoder
                self.input_blocks_T1.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        ch *= 3
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        model_channels *= 3
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = 288

        self.input_blocks_2 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_3 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_4 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_5 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_6 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_T1 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])

        self.SE_Attention_com = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_2 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_3 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_4 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_5 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_6 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_T1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)

        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 5 * int(conv_ch / 2), 3 * conv_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch * 3, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, low_res, other):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h1 = th.stack([x[:,0,:,:], low_res[:,0,:,:]], dim=1).type(self.dtype)
        h2 = th.stack([x[:,1,:,:], low_res[:,1,:,:]], dim=1).type(self.dtype)
        h3 = th.stack([x[:,2,:,:], low_res[:,2,:,:]], dim=1).type(self.dtype)
        # h1 = x[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        # h2 = x[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        # h3 = x[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        # h4 = low_res[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        # h5 = low_res[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        # h6 = low_res[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        # h1 = x.type(self.dtype)
        # h2 = low_res.type(self.dtype)

        # print(th.count_nonzero(other))
        # has_other = (th.count_nonzero(other) > 0)
        ht = other.type(self.dtype)
        # print(ht.shape)

        for idx in range(len(self.input_blocks)):
            h1 = self.input_blocks[idx](h1, emb)
            h2 = self.input_blocks_2[idx](h2, emb)
            h3 = self.input_blocks_3[idx](h3, emb)
            # h4 = self.input_blocks_4[idx](h4, emb)
            # h5 = self.input_blocks_5[idx](h5, emb)
            # h6 = self.input_blocks_6[idx](h6, emb)
            ht = self.input_blocks_T1[idx](ht, emb)
            # print(h1.shape, h2.shape, ht.shape)
            hs.append((1 / 4) * h1 + (1 / 4) * h2 + (1 / 4) * h3 + (1 / 4) * ht)

        com_h1 = self.conv_common(h1)
        com_h2 = self.conv_common(h2)
        com_h3 = self.conv_common(h3)
        # com_h4 = self.conv_common(h4)
        # com_h5 = self.conv_common(h5)
        # com_h6 = self.conv_common(h6)

        dist_h1 = self.conv_distinct(h1)
        dist_h2 = self.conv_distinct(h2)
        dist_h3 = self.conv_distinct(h3)
        # dist_h4 = self.conv_distinct(h4)
        # dist_h5 = self.conv_distinct(h5)
        # dist_h6 = self.conv_distinct(h6)

        dist_h1 = self.SE_Attention_dist_1(dist_h1)
        dist_h2 = self.SE_Attention_dist_2(dist_h2)
        dist_h3 = self.SE_Attention_dist_3(dist_h3)
        # dist_h4 = self.SE_Attention_dist_4(dist_h4)
        # dist_h5 = self.SE_Attention_dist_5(dist_h5)
        # dist_h6 = self.SE_Attention_dist_6(dist_h6)


        com_ht = self.conv_common(ht)
        dist_ht = self.conv_distinct(ht)
        com_h = self.SE_Attention_com((1 / 4) * com_h1 + (1 / 4) * com_h2 + (1 / 4) * com_h3 + (1 / 4) * com_ht)
        dist_ht = self.SE_Attention_dist_3(dist_ht)
        h = th.cat([com_h, dist_h1, dist_h2, dist_h3, dist_ht], dim=1)
        h = self.dim_reduction_non_zeros(h)

        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        return com_h1, com_h2, com_h3, com_ht, dist_h1, dist_h2, dist_h3, dist_ht, self.out(h)



class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            pool="adaptive",
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
        


class UNetb2Model(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        ch *= 2
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        model_channels *= 2
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = 288

        self.input_blocks_2 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_3 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_4 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_5 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_6 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_T1 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])

        self.SE_Attention_com = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_2 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_3 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_4 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_5 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_6 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_T1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)

        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 5 * int(conv_ch / 2), 2 * conv_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch * 2, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, low_res, other=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h1 = x[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        h2 = x[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        # h3 = x[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        h3 = low_res[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        h4 = low_res[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        # h6 = low_res[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        # h1 = x.type(self.dtype)
        # h2 = low_res.type(self.dtype)

        # print(th.count_nonzero(other))
        # has_other = (th.count_nonzero(other) > 0)
        # ht = other.type(self.dtype)
        # print(ht.shape)

        for idx in range(len(self.input_blocks)):
            h1 = self.input_blocks[idx](h1, emb)
            h2 = self.input_blocks_2[idx](h2, emb)
            h3 = self.input_blocks_3[idx](h3, emb)
            h4 = self.input_blocks_4[idx](h4, emb)
            # h5 = self.input_blocks_5[idx](h5, emb)
            # h6 = self.input_blocks_6[idx](h6, emb)
            # ht = self.input_blocks_T1[idx](ht, emb)
            # print(h1.shape, h2.shape, h3.shape, h4.shape)
            hs.append((1 / 4) * h1 + (1 / 4) * h2 + (1 / 4) * h3 + (1 / 4) * h4 )
        # print(h1.shape, h2.shape, h3.shape, h4.shape)
        com_h1 = self.conv_common(h1)
        com_h2 = self.conv_common(h2)
        com_h3 = self.conv_common(h3)
        com_h4 = self.conv_common(h4)
        # com_h5 = self.conv_common(h5)
        # com_h6 = self.conv_common(h6)

        dist_h1 = self.conv_distinct(h1)
        dist_h2 = self.conv_distinct(h2)
        dist_h3 = self.conv_distinct(h3)
        dist_h4 = self.conv_distinct(h4)
        # dist_h5 = self.conv_distinct(h5)
        # dist_h6 = self.conv_distinct(h6)

        dist_h1 = self.SE_Attention_dist_1(dist_h1)
        dist_h2 = self.SE_Attention_dist_2(dist_h2)
        dist_h3 = self.SE_Attention_dist_3(dist_h3)
        dist_h4 = self.SE_Attention_dist_4(dist_h4)
        # dist_h5 = self.SE_Attention_dist_5(dist_h5)
        # dist_h6 = self.SE_Attention_dist_6(dist_h6)


        # com_ht = self.conv_common(ht)
        # dist_ht = self.conv_distinct(ht)
        com_h = self.SE_Attention_com((1 / 4) * com_h1 + (1 / 4) * com_h2 + (1 / 4) * com_h3 + (1 / 4) * com_h4)
        # dist_ht = self.SE_Attention_dist_3(dist_ht)
        h = th.cat([com_h, dist_h1, dist_h2, dist_h3, dist_h4], dim=1)
        h = self.dim_reduction_non_zeros(h)
        # print(h.shape)
        h = self.middle_block(h, emb)
        # print(h.shape)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            # print(h.shape)
            h = module(h, emb)
        h = h.type(x.dtype)

        return com_h1, com_h2, com_h3, com_h4, dist_h1, dist_h2, dist_h3, dist_h4, self.out(h)



class UNetb2t1Model(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        ch *= 2
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        model_channels *= 2
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        conv_ch = 288

        self.input_blocks_2 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_3 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_4 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_5 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        # self.input_blocks_6 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])
        self.input_blocks_T1 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks])

        self.SE_Attention_com = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_2 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_3 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_4 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_5 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_6 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_T1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)

        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 6 * int(conv_ch / 2), 2 * conv_ch, 1, padding=0),
            nn.SiLU()
        )

        self.conv_common = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.conv_distinct = nn.Sequential(
            conv_nd(dims, conv_ch, int(conv_ch / 2), 3, padding=1),
            nn.SiLU()
        )

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch * 2, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, low_res, other=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h1 = x[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        h2 = x[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        # h3 = x[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        h3 = low_res[:,0,:,:].unsqueeze(dim=1).type(self.dtype)
        h4 = low_res[:,1,:,:].unsqueeze(dim=1).type(self.dtype)
        # h6 = low_res[:,2,:,:].unsqueeze(dim=1).type(self.dtype)
        # h1 = x.type(self.dtype)
        # h2 = low_res.type(self.dtype)

        # print(th.count_nonzero(other))
        # has_other = (th.count_nonzero(other) > 0)
        ht = other.type(self.dtype)
        # print(ht.shape)

        for idx in range(len(self.input_blocks)):
            h1 = self.input_blocks[idx](h1, emb)
            h2 = self.input_blocks_2[idx](h2, emb)
            h3 = self.input_blocks_3[idx](h3, emb)
            h4 = self.input_blocks_4[idx](h4, emb)
            # h5 = self.input_blocks_5[idx](h5, emb)
            # h6 = self.input_blocks_6[idx](h6, emb)
            ht = self.input_blocks_T1[idx](ht, emb)
            # print(h1.shape, h2.shape, h3.shape, h4.shape)
            hs.append((1 / 4) * h1 + (1 / 4) * h2 + (1 / 4) * h3 + (1 / 4) * h4 )
        # print(h1.shape, h2.shape, h3.shape, h4.shape)
        com_h1 = self.conv_common(h1)
        com_h2 = self.conv_common(h2)
        com_h3 = self.conv_common(h3)
        com_h4 = self.conv_common(h4)
        # com_h5 = self.conv_common(h5)
        # com_h6 = self.conv_common(h6)

        dist_h1 = self.conv_distinct(h1)
        dist_h2 = self.conv_distinct(h2)
        dist_h3 = self.conv_distinct(h3)
        dist_h4 = self.conv_distinct(h4)
        # dist_h5 = self.conv_distinct(h5)
        # dist_h6 = self.conv_distinct(h6)

        dist_h1 = self.SE_Attention_dist_1(dist_h1)
        dist_h2 = self.SE_Attention_dist_2(dist_h2)
        dist_h3 = self.SE_Attention_dist_3(dist_h3)
        dist_h4 = self.SE_Attention_dist_4(dist_h4)
        # dist_h5 = self.SE_Attention_dist_5(dist_h5)
        # dist_h6 = self.SE_Attention_dist_6(dist_h6)


        com_ht = self.conv_common(ht)
        dist_ht = self.conv_distinct(ht)
        com_h = self.SE_Attention_com((1 / 5) * com_h1 + (1 / 5) * com_h2 + (1 / 5) * com_h3 + (1 / 5) * com_h4 + (1 / 5) * com_ht)
        dist_ht = self.SE_Attention_dist_T1(dist_ht)
        h = th.cat([com_h, dist_h1, dist_h2, dist_h3, dist_h4, dist_ht], dim=1)
        h = self.dim_reduction_non_zeros(h)
        # print(h.shape)
        h = self.middle_block(h, emb)
        # print(h.shape)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            # print(h.shape)
            h = module(h, emb)
        h = h.type(x.dtype)

        return com_h1, com_h2, com_h3, com_h4, com_ht, dist_h1, dist_h2, dist_h3, dist_h4, dist_ht, self.out(h)




class SuperResUNetb2t1(UNetb2t1Model):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels, *args, **kwargs)
        print(image_size)
    def forward(self, x, timesteps,**kwargs):
        # _, _, new_height, new_width = x.shape
        # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(x, timesteps, kwargs['low_res'], kwargs['other'])
    
class SuperResUNetb2(UNetb2Model):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels, *args, **kwargs)
        print(image_size)
    def forward(self, x, timesteps,**kwargs):
        # _, _, new_height, new_width = x.shape
        # upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(x, timesteps, kwargs['low_res'], kwargs['other'])
