from .unet_partial import *


class UNetb2d6t1Model(nn.Module):
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
        self.input_blocks_t1 =  nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, 1, ch, 3, padding=1))]
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
                self.input_blocks_t1.append(TimestepEmbedSequential(*layers))
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
                self.input_blocks_t1.append(
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
        self.input_blocks_T1 = nn.ModuleList([copy.deepcopy(module) for module in self.input_blocks_t1])

        self.SE_Attention_com = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_2 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_3 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        # self.SE_Attention_dist_4 = SE_Attention(channel=int(conv_ch / 2), reduction=8)
        self.SE_Attention_dist_T1 = SE_Attention(channel=int(conv_ch / 2), reduction=8)

        self.dim_reduction_non_zeros = nn.Sequential(
            conv_nd(dims, 4 * int(conv_ch / 2), 2 * conv_ch, 1, padding=0),
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

    def forward(self, x, timesteps, low_res, other=None):
        hs = []

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h1 = x.type(self.dtype)
        h2 = low_res.type(self.dtype)
        ht = other.type(self.dtype)

        for idx in range(len(self.input_blocks)):
            h1 = self.input_blocks[idx](h1, emb)
            h2 = self.input_blocks_2[idx](h2, emb)
            ht = self.input_blocks_T1[idx](ht, emb)

            hs.append((1 / 2) * h1 + (1 / 2) * h2)

        com_h1 = self.conv_common(h1)
        com_h2 = self.conv_common(h2)
        dist_h1 = self.conv_distinct(h1)
        dist_h2 = self.conv_distinct(h2)

        dist_h1 = self.SE_Attention_dist_1(dist_h1)
        dist_h2 = self.SE_Attention_dist_2(dist_h2)

        com_ht = self.conv_common(ht)
        dist_ht = self.conv_distinct(ht)
        com_h = self.SE_Attention_com((1 / 3) * com_h1 + (1 / 3) * com_h2 + (1 / 3) * com_ht)
        dist_ht = self.SE_Attention_dist_T1(dist_ht)
        h = th.cat([com_h, dist_h1, dist_h2, dist_ht], dim=1)
        h = self.dim_reduction_non_zeros(h)

        h = self.middle_block(h, emb)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            # print(h.shape)
            h = module(h, emb)
        h = h.type(x.dtype)

        return com_h1, com_h2, com_ht, dist_h1, dist_h2, dist_ht, self.out(h)
    

class SuperResUNetd6t1(UNetb2d6t1Model):
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



