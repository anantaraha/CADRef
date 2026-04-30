import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

EPS = 1e-12
LAMBDA_UNCERT = 1.0
PRINT_STATE_STATS = True


class Rel:
    def __init__(self, model, args, device):
        self.model = model
        self.device = device
        self.args = args

        self.prototypes = None      # [C, D]
        self.class_var = None       # [C]
        self.class_uncert = None    # [C]

    def set_state(self, prototypes, class_var, class_uncert):
        self.prototypes = prototypes
        self.class_var = class_var
        self.class_uncert = class_uncert

    def _entropy(self, logits):
        p = F.softmax(logits, dim=1)
        return -(p * torch.log(p.clamp_min(EPS))).sum(dim=1)

    @torch.no_grad()
    def get_state(self, features, logits):
        """
        features[c]: [Nc, D], grouped by predicted class from get_features()
        logits[c]:   [Nc, C], grouped by predicted class from get_features()
        """
        prototypes = []
        class_var = []
        class_uncert = []

        for c in range(len(features)):
            feat_c = features[c]   # [Nc, D]
            logit_c = logits[c]    # [Nc, C]

            if feat_c.numel() == 0:
                raise ValueError(f"Empty feature group for class {c}")

            mu_c = feat_c.mean(dim=0)  # prototype: [D]

            # sigma_k^2 = average squared distance from prototype
            diff = feat_c - mu_c.unsqueeze(0)
            sigma2_c = diff.pow(2).sum(dim=1).mean().clamp_min(EPS)

            # U_k = average entropy of samples assigned to class k
            entropy_c = self._entropy(logit_c).mean()

            prototypes.append(mu_c)
            class_var.append(sigma2_c)
            class_uncert.append(entropy_c)

        prototypes = torch.stack(prototypes)       # [C, D]
        class_var = torch.stack(class_var)         # [C]
        class_uncert = torch.stack(class_uncert)   # [C]

        if PRINT_STATE_STATS:
            print("\n[Rel state statistics]")
            print(f"class_var:    mean={class_var.mean().item():.6f}, min={class_var.min().item():.6f}, max={class_var.max().item():.6f}")
            print(f"class_uncert: mean={class_uncert.mean().item():.6f}, min={class_uncert.min().item():.6f}, max={class_uncert.max().item():.6f}")
            print(f"lambda*U:     mean={(LAMBDA_UNCERT * class_uncert).mean().item():.6f}, "
                f"min={(LAMBDA_UNCERT * class_uncert).min().item():.6f}, "
                f"max={(LAMBDA_UNCERT * class_uncert).max().item():.6f}")

        return prototypes, class_var, class_uncert

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        prototypes = self.prototypes.to(self.device)        # [C, D]
        class_var = self.class_var.to(self.device)          # [C]
        class_uncert = self.class_uncert.to(self.device)    # [C]

        printed_eval_stats = False

        for images, _ in tqdm(data_loader):
            images = images.to(self.device)

            logits, feat = self.model.get_feature(images)   # feat: [B, D]
            assigned = logits.argmax(dim=1)                 # predicted class \tilde{y}

            # ||f_i - mu_k||^2 for all k
            diff = feat.unsqueeze(1) - prototypes.unsqueeze(0)   # [B, C, D]
            sq_dist = diff.pow(2).sum(dim=2)                     # [B, C]

            # d_NA(i,k) = ||f_i - mu_k||^2 / sigma_k^2 + U_k
            base_term = sq_dist / class_var.unsqueeze(0).clamp_min(EPS)
            uncert_term = LAMBDA_UNCERT * class_uncert.unsqueeze(0)
            d_na = base_term + uncert_term

            if PRINT_STATE_STATS and not printed_eval_stats:
                pred_base = base_term[torch.arange(feat.size(0), device=self.device), assigned]
                pred_uncert = uncert_term[0, assigned]

                print("\n[Rel eval statistics: first batch, assigned class]")
                print(f"base term:    mean={pred_base.mean().item():.6f}, min={pred_base.min().item():.6f}, max={pred_base.max().item():.6f}")
                print(f"lambda*U:     mean={pred_uncert.mean().item():.6f}, min={pred_uncert.min().item():.6f}, max={pred_uncert.max().item():.6f}")
                print(f"ratio U/base: mean={(pred_uncert / pred_base.clamp_min(EPS)).mean().item():.6f}")
                printed_eval_stats = True

            # r_i = exp(-d_i,y) / sum_k exp(-d_i,k)
            rel_all = F.softmax(-d_na, dim=1)                    # [B, C]
            rel_score = rel_all[torch.arange(feat.size(0), device=self.device), assigned]

            result.append(rel_score.cpu().numpy())

        return np.concatenate(result)