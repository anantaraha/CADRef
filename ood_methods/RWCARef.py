import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

EPS = 1e-12
PRINT_STATE_STATS = True

BETA_H = 1.0
USE_DISTANCE_WEIGHT = False
USE_ENTROPY_WEIGHT = True


class RWCARef:
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device

        self.weighted_mean = None
        self.class_var = None
        self.class_uncert = None
        self.class_uncert_norm = None

    def set_state(self, weighted_mean, class_var, class_uncert, class_uncert_norm):
        self.weighted_mean = weighted_mean
        self.class_var = class_var
        self.class_uncert = class_uncert
        self.class_uncert_norm = class_uncert_norm

    def _entropy(self, logits):
        p = F.softmax(logits, dim=1)
        return -(p * torch.log(p.clamp_min(EPS))).sum(dim=1)

    @torch.no_grad()
    def get_state(self, features, logits):
        """
        features[c]: [Nc, D], grouped by predicted class
        logits[c]:   [Nc, C], grouped by predicted class
        """

        # ---------- First pass: ordinary prototypes, spread, uncertainty ----------
        mean_list = []
        var_list = []
        uncert_list = []

        for c in range(len(features)):
            feat_c = features[c]   # [Nc, D]
            logit_c = logits[c]    # [Nc, C]

            if feat_c.numel() == 0:
                raise ValueError(f"Empty feature group for class {c}")

            mu_c = feat_c.mean(dim=0)

            diff = feat_c - mu_c.unsqueeze(0)
            sigma2_c = diff.pow(2).sum(dim=1).mean().clamp_min(EPS)

            uncert_c = self._entropy(logit_c).mean()

            mean_list.append(mu_c)
            var_list.append(sigma2_c)
            uncert_list.append(uncert_c)

        mean = torch.stack(mean_list)               # [C, D]
        class_var = torch.stack(var_list)           # [C]
        class_uncert = torch.stack(uncert_list)     # [C]

        # normalized U_c
        u_min = class_uncert.min()
        u_max = class_uncert.max()
        class_uncert_norm = (class_uncert - u_min) / (u_max - u_min + EPS)

        # ---------- Normalize sample-level entropy across all training samples ----------
        all_entropy = []
        for c in range(len(logits)):
            logit_c = logits[c]
            if logit_c.numel() > 0:
                all_entropy.append(self._entropy(logit_c))

        all_entropy = torch.cat(all_entropy)
        h_min = all_entropy.min()
        h_max = all_entropy.max()

        # ---------- Second pass: reliability-weighted prototypes ----------
        weighted_mean_list = []

        all_H_norm = []
        all_B = []
        all_R = []

        for c in range(len(features)):
            feat_c = features[c]   # [Nc, D]
            mu_c = mean[c]         # [D]

            # B_i = ||f_i - mu_c||^2 / sigma_c^2
            diff = feat_c - mu_c.unsqueeze(0)
            B_i = diff.pow(2).sum(dim=1) / class_var[c].clamp_min(EPS)  # [Nc]

            # Sample-entropy-only reliability:
            # R_i = exp(-beta_H * H_norm_i)
            H_i = self._entropy(logits[c])  # [Nc]
            H_norm_i = (H_i - h_min) / (h_max - h_min + EPS)

            R_i = torch.exp(-BETA_H * H_norm_i)

            weighted_mu_c = (R_i.unsqueeze(1) * feat_c).sum(dim=0) / R_i.sum().clamp_min(EPS)

            weighted_mean_list.append(weighted_mu_c)

            all_B.append(B_i)
            all_R.append(R_i)
            all_H_norm.append(H_norm_i)

        weighted_mean = torch.stack(weighted_mean_list)  # [C, D]

        if PRINT_STATE_STATS:
            all_B = torch.cat(all_B)
            all_R = torch.cat(all_R)

            print("\n[RWCARef state statistics]")
            print(f"class_var:        mean={class_var.mean().item():.6f}, min={class_var.min().item():.6f}, max={class_var.max().item():.6f}")
            print(f"class_uncert U:   mean={class_uncert.mean().item():.6f}, min={class_uncert.min().item():.6f}, max={class_uncert.max().item():.6f}")
            print(f"U_norm:           mean={class_uncert_norm.mean().item():.6f}, min={class_uncert_norm.min().item():.6f}, max={class_uncert_norm.max().item():.6f}")
            print(f"B_i:              mean={all_B.mean().item():.6f}, min={all_B.min().item():.6f}, max={all_B.max().item():.6f}")
            
            all_H_norm = torch.cat(all_H_norm)
            print(f"H_norm_i:         mean={all_H_norm.mean().item():.6f}, min={all_H_norm.min().item():.6f}, max={all_H_norm.max().item():.6f}")
            print(f"R_i:              mean={all_R.mean().item():.6f}, min={all_R.min().item():.6f}, max={all_R.max().item():.6f}")
            print(f"BETA_H={BETA_H}, USE_DISTANCE_WEIGHT={USE_DISTANCE_WEIGHT}, USE_ENTROPY_WEIGHT={USE_ENTROPY_WEIGHT}")

            all_B_excess = torch.clamp(all_B - 1.0, min=0.0)
            print(f"B_excess:         mean={all_B_excess.mean().item():.6f}, min={all_B_excess.min().item():.6f}, max={all_B_excess.max().item():.6f}")

        return weighted_mean, class_var, class_uncert, class_uncert_norm

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        weighted_mean = self.weighted_mean.to(self.device)  # [C, D]

        printed_eval_stats = False

        for images, _ in tqdm(data_loader):
            images = images.to(self.device)

            logits, feat = self.model.get_feature(images)
            class_ids = torch.argmax(logits, dim=1)

            tm = weighted_mean[class_ids]  # [B, D]

            error = (feat - tm).abs().sum(dim=1) / feat.abs().sum(dim=1).clamp_min(EPS)

            if PRINT_STATE_STATS and not printed_eval_stats:
                print("\n[RWCARef eval statistics: first batch]")
                print(f"CARef error: mean={error.mean().item():.6f}, min={error.min().item():.6f}, max={error.max().item():.6f}")
                printed_eval_stats = True

            score = -error
            result.append(score.cpu().numpy())

        return np.concatenate(result)