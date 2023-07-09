from lavi.utils.lib import *
from lavi.models.visbackbone.video_swin import get_vidswin_model

class EncVideo(T.nn.Module):

    def __init__(self, 
                 max_size_frame,
                 max_size_patch,
                 size_img,
                 vis_backbone_size,
                 vis_backbone_init,
                 kinetics,
                 hidden_size):
            
        super().__init__()
        self.swin = get_vidswin_model(
                        size_img,
                        vis_backbone_size,
                        vis_backbone_init,
                        kinetics)
        self.latent_feat_size = self.swin.norm.normalized_shape[0]
        self.img_feature_dim = hidden_size
        self.swinbert = False
        self.max_size_frame = max_size_frame 
        self.max_size_patch = max_size_patch 

        if not self.swinbert:
            if self.latent_feat_size != self.img_feature_dim:
                self.fc = T.nn.Linear(
                    self.latent_feat_size, self.img_feature_dim)
            else:
                self.fc = None
            self.emb_cls = T.nn.Parameter(
                    0.02*T.randn(1, 1, 1, self.img_feature_dim))
            self.emb_pos = T.nn.Parameter(
                0.02*T.randn(
                    1, 1, 1+self.max_size_patch**2, self.img_feature_dim))
            self.emb_len = T.nn.Parameter(
                0.02*T.randn(
                    1, self.max_size_frame, 1, self.img_feature_dim))
            self.emb_odr = T.nn.Parameter(
                0.02*T.randn(1, 1, 1, self.img_feature_dim))
            self.norm = T.nn.LayerNorm(self.img_feature_dim)
        else:
            self.fc = T.nn.Linear(self.latent_feat_size, 512)
            self.img_embedding = T.nn.Linear(512, self.img_feature_dim)
        self.transform_normalize = None

    def forward(self, img, odr=None, vt_mask=None):
        _B, _T, _C, _H, _W = img.shape
        _h, _w = _H//32, _W//32

        if self.transform_normalize is not None:
            img = self.transform_normalize(img)

        f_img = self.swin(img.transpose(1, 2)).transpose(1, 2)
        f_img = f_img.permute(0, 1, 3, 4, 2).view(
            [_B, _T, _h*_w, self.latent_feat_size])

        if self.fc is not None:
            f_img = self.fc(f_img)

        # for swinbert initialized
        if self.swinbert:
            f_img = self.img_embedding(f_img)
            fake_cls_token = T.zeros(
                (_B, _T, 1, self.img_feature_dim), dtype=f_img.dtype,
                device=f_img.device)
            f_img = T.cat([fake_cls_token, f_img], dim=2)

            m_img = T.ones(_h*_w).long().cuda().unsqueeze(0).unsqueeze(0)
            m_img = m_img.expand([_B, _T, -1]).contiguous()
            fake_cls_mask = T.zeros((_B, _T, 1), dtype=m_img.dtype,
                                    device=m_img.device)
            m_img = T.cat([fake_cls_mask, m_img], dim=2)

            f_img = f_img.view([_B, _T*(1+_h*_w), -1])
            m_img = m_img.view([_B, _T*(1+_h*_w)])
            return f_img, m_img

        f_img = T.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_img], dim=2)
        f_img += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1+_h*_w, :]

        if odr is not None:
            emb_len = []  # feed order
            for b in range(_B):
                tmp = T.cat([
                    self.emb_len[:, i:i+1, :, :]
                    if i == p else self.emb_odr
                    for i, p in enumerate(odr[b])], dim=1)
                emb_len.append(tmp)
            emb_len = T.cat(emb_len, dim=0)
            f_img += emb_len

        else:
            f_img += self.emb_len.expand([_B, -1, 1+_h*_w, -1])[:, :_T, :, :]

        f_img = self.norm(f_img).view([_B, _T*(1+_h*_w), -1])

        m_img = T.ones(1+_h*_w).long().cuda().unsqueeze(0).unsqueeze(0)
        m_img = m_img.expand([_B, _T, -1]).contiguous()
        if vt_mask is not None:
            m_img = m_img * vt_mask
        m_img = m_img.view([_B, _T*(1+_h*_w)])

        return f_img, m_img

    def load_vis_ckpt_from_lavender(self, loaded_state_dict):
        model_keys = set([k for k in list(self.state_dict().keys())])
        load_keys = set(loaded_state_dict.keys())

        for k in load_keys:
            if k.startswith('enc_img.'):
                loaded_state_dict[k[len('enc_img.'):]] = loaded_state_dict[k]
                del loaded_state_dict[k]
        load_keys = set(loaded_state_dict.keys())

        toload = {}
        mismatched_shape_keys = []
        for k in model_keys:
            if k in load_keys:
                if self.state_dict()[k].shape != loaded_state_dict[k].shape:
                    mismatched_shape_keys.append(
                        (k, loaded_state_dict[k].shape,
                         self.state_dict()[k].shape))
                else:
                    toload[k] = loaded_state_dict[k]

        print("You can ignore the keys with `position_ids` or from task heads")
        strct_loading = True
        unexpected = load_keys.difference(model_keys)
        if len(unexpected):
            strct_loading = False
            print("=========================Unexpected==================================")
            print(f"\tIn total {len(unexpected)}, {sorted(unexpected)}")

        missing = model_keys.difference(load_keys)
        if len(missing):
            strct_loading = False
            print("===========================Missing===================================")
            print(f"\tIn total {len(missing)}, {sorted(missing)}")

        if len(mismatched_shape_keys):
            strct_loading = False
            print("======================Shape Mismatched===============================")
            print(f"\tIn total {len(mismatched_shape_keys)}, "
                  f"{sorted(mismatched_shape_keys)}")


        self.load_state_dict(toload, strict=strct_loading)
