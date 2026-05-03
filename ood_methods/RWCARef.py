import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

EPS = 1e-12


class RWCARef:
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device

        self.beta_b = args.rw_beta_b
        self.beta_h = args.rw_beta_h
        self.use_distance_weight = args.rw_use_distance
        self.use_entropy_weight = args.rw_use_entropy
        self.print_stats = args.rw_print_stats

        self.weighted_mean = None

    def set_state(self, weighted_mean):
        self.weighted_mean = weighted_mean

    def _entropy(self, logits):
        p = F.softmax(logits, dim=1)
        return -(p * torch.log(p.clamp_min(EPS))).sum(dim=1)

    @torch.no_grad()
    def get_state(self, features, logits):
        """
        features[c]: [Nc, D], grouped by predicted class
        logits[c]:   [Nc, C], grouped by predicted class

        Builds reliability-weighted class references.
        """

        num_classes = len(features)
        log_k = torch.log(torch.tensor(float(num_classes), device=features[0].device)).clamp_min(EPS)

        # ---------- First pass: ordinary class means and spreads ----------
        mean_list = []
        var_list = []

        for c in range(num_classes):
            feat_c = features[c]

            if feat_c.numel() == 0:
                raise ValueError(f"Empty feature group for class {c}")

            mu_c = feat_c.mean(dim=0)

            diff = feat_c - mu_c.unsqueeze(0)
            sigma2_c = diff.pow(2).sum(dim=1).mean().clamp_min(EPS)

            mean_list.append(mu_c)
            var_list.append(sigma2_c)

        class_mean = torch.stack(mean_list)   # [C, D]
        class_var = torch.stack(var_list)     # [C]

        # ---------- Second pass: sample reliability and weighted references ----------
        weighted_mean_list = []

        all_B = []
        all_B_excess = []
        all_H_norm = []
        all_R = []
        all_R_dist = []
        all_R_ent = []

        for c in range(num_classes):
            feat_c = features[c]       # [Nc, D]
            logit_c = logits[c]        # [Nc, C]
            mu_c = class_mean[c]       # [D]

            # B_i = ||f_i - mu_c||^2 / (sigma_c^2 + eps)
            diff = feat_c - mu_c.unsqueeze(0)
            B_i = diff.pow(2).sum(dim=1) / class_var[c].clamp_min(EPS)

            # B_i^+ = max(0, B_i - 1)
            B_excess = torch.clamp(B_i - 1.0, min=0.0)

            # H_i_norm = H(p_i) / log(K)
            H_i = self._entropy(logit_c)
            H_norm_i = H_i / log_k

            # Optional terms
            if self.use_distance_weight:
                R_dist = torch.exp(-self.beta_b * B_excess)
            else:
                R_dist = torch.ones_like(B_i)

            if self.use_entropy_weight:
                R_ent = torch.exp(-self.beta_h * H_norm_i)
            else:
                R_ent = torch.ones_like(B_i)

            # R_i = exp(-beta_B B_i^+) * exp(-beta_H H_i_norm)
            R_i = R_dist * R_ent

            weighted_mu_c = (R_i.unsqueeze(1) * feat_c).sum(dim=0) / R_i.sum().clamp_min(EPS)
            weighted_mean_list.append(weighted_mu_c)

            all_B.append(B_i)
            all_B_excess.append(B_excess)
            all_H_norm.append(H_norm_i)
            all_R.append(R_i)
            all_R_dist.append(R_dist)
            all_R_ent.append(R_ent)

        weighted_mean = torch.stack(weighted_mean_list)  # [C, D]

        if self.print_stats:
            all_B = torch.cat(all_B)
            all_B_excess = torch.cat(all_B_excess)
            all_H_norm = torch.cat(all_H_norm)
            all_R = torch.cat(all_R)
            all_R_dist = torch.cat(all_R_dist)
            all_R_ent = torch.cat(all_R_ent)

            print("\n[RWCARef state statistics]")
            print(f"beta_b={self.beta_b}, beta_h={self.beta_h}")
            print(f"use_distance_weight={self.use_distance_weight}, use_entropy_weight={self.use_entropy_weight}")

            print(f"class_var:  mean={class_var.mean().item():.6f}, min={class_var.min().item():.6f}, max={class_var.max().item():.6f}")
            print(f"B_i:        mean={all_B.mean().item():.6f}, min={all_B.min().item():.6f}, max={all_B.max().item():.6f}")
            print(f"B_excess:   mean={all_B_excess.mean().item():.6f}, min={all_B_excess.min().item():.6f}, max={all_B_excess.max().item():.6f}")
            print(f"H_norm_i:   mean={all_H_norm.mean().item():.6f}, min={all_H_norm.min().item():.6f}, max={all_H_norm.max().item():.6f}")
            print(f"R_dist:     mean={all_R_dist.mean().item():.6f}, min={all_R_dist.min().item():.6f}, max={all_R_dist.max().item():.6f}")
            print(f"R_ent:      mean={all_R_ent.mean().item():.6f}, min={all_R_ent.min().item():.6f}, max={all_R_ent.max().item():.6f}")
            print(f"R_i:        mean={all_R.mean().item():.6f}, min={all_R.min().item():.6f}, max={all_R.max().item():.6f}")

        return weighted_mean

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        weighted_mean = self.weighted_mean.to(self.device)

        printed_eval_stats = False

        for images, _ in tqdm(data_loader):
            images = images.to(self.device)

            logits, feat = self.model.get_feature(images)
            class_ids = torch.argmax(logits, dim=1)

            tm = weighted_mean[class_ids]

            error = (feat - tm).abs().sum(dim=1) / feat.abs().sum(dim=1).clamp_min(EPS)

            if self.print_stats and not printed_eval_stats:
                print("\n[RWCARef eval statistics: first batch]")
                print(f"CARef error: mean={error.mean().item():.6f}, min={error.min().item():.6f}, max={error.max().item():.6f}")
                printed_eval_stats = True

            score = -error
            result.append(score.cpu().numpy())

        return np.concatenate(result)
