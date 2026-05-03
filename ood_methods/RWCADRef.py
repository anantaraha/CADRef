import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.linear_mapping import get_linear_layer_mapping

EPS = 1e-12


class RWCADRef:
    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device
        self.logit_method = args.logit_method
        self.linear = get_linear_layer_mapping(args.model, args.ind_dataset, model)

        self.beta_b = args.rw_beta_b
        self.beta_h = args.rw_beta_h
        self.use_distance_weight = args.rw_use_distance
        self.use_entropy_weight = args.rw_use_entropy
        self.print_stats = args.rw_print_stats

        self.train_mean = None
        self.global_mean_logit_score = None

    def set_state(self, train_mean, global_mean_logit_score):
        self.train_mean = train_mean
        self.global_mean_logit_score = global_mean_logit_score

    def _entropy(self, logits):
        p = F.softmax(logits, dim=1)
        return -(p * torch.log(p.clamp_min(EPS))).sum(dim=1)

    @torch.no_grad()
    def get_global_mean_logit_score(self, logits):
        train_logit_score = []
        for i in range(len(logits)):
            train_logit_score.append(self.get_logits_score(logits[i]))
        return torch.mean(torch.cat(train_logit_score), dim=0)

    @torch.no_grad()
    def get_state(self, features, logits):
        num_classes = len(features)
        log_k = torch.log(torch.tensor(float(num_classes), device=features[0].device)).clamp_min(EPS)

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

        class_mean = torch.stack(mean_list)
        class_var = torch.stack(var_list)

        weighted_mean_list = []

        all_B, all_B_excess, all_H_norm = [], [], []
        all_R, all_R_dist, all_R_ent = [], [], []

        for c in range(num_classes):
            feat_c = features[c]
            logit_c = logits[c]
            mu_c = class_mean[c]

            diff = feat_c - mu_c.unsqueeze(0)
            B_i = diff.pow(2).sum(dim=1) / class_var[c].clamp_min(EPS)
            B_excess = torch.clamp(B_i - 1.0, min=0.0)

            H_i = self._entropy(logit_c)
            H_norm_i = H_i / log_k

            if self.use_distance_weight:
                R_dist = torch.exp(-self.beta_b * B_excess)
            else:
                R_dist = torch.ones_like(B_i)

            if self.use_entropy_weight:
                R_ent = torch.exp(-self.beta_h * H_norm_i)
            else:
                R_ent = torch.ones_like(B_i)

            R_i = R_dist * R_ent

            weighted_mu_c = (R_i.unsqueeze(1) * feat_c).sum(dim=0) / R_i.sum().clamp_min(EPS)
            weighted_mean_list.append(weighted_mu_c)

            all_B.append(B_i)
            all_B_excess.append(B_excess)
            all_H_norm.append(H_norm_i)
            all_R.append(R_i)
            all_R_dist.append(R_dist)
            all_R_ent.append(R_ent)

        train_mean = torch.stack(weighted_mean_list)
        global_mean_logit_score = self.get_global_mean_logit_score(logits)

        if self.print_stats:
            all_B = torch.cat(all_B)
            all_B_excess = torch.cat(all_B_excess)
            all_H_norm = torch.cat(all_H_norm)
            all_R = torch.cat(all_R)
            all_R_dist = torch.cat(all_R_dist)
            all_R_ent = torch.cat(all_R_ent)

            print("\n[RWCADRef state statistics]")
            print(f"beta_b={self.beta_b}, beta_h={self.beta_h}")
            print(f"use_distance_weight={self.use_distance_weight}, use_entropy_weight={self.use_entropy_weight}")
            print(f"class_var:  mean={class_var.mean().item():.6f}, min={class_var.min().item():.6f}, max={class_var.max().item():.6f}")
            print(f"B_i:        mean={all_B.mean().item():.6f}, min={all_B.min().item():.6f}, max={all_B.max().item():.6f}")
            print(f"B_excess:   mean={all_B_excess.mean().item():.6f}, min={all_B_excess.min().item():.6f}, max={all_B_excess.max().item():.6f}")
            print(f"H_norm_i:   mean={all_H_norm.mean().item():.6f}, min={all_H_norm.min().item():.6f}, max={all_H_norm.max().item():.6f}")
            print(f"R_dist:     mean={all_R_dist.mean().item():.6f}, min={all_R_dist.min().item():.6f}, max={all_R_dist.max().item():.6f}")
            print(f"R_ent:      mean={all_R_ent.mean().item():.6f}, min={all_R_ent.min().item():.6f}, max={all_R_ent.max().item():.6f}")
            print(f"R_i:        mean={all_R.mean().item():.6f}, min={all_R.min().item():.6f}, max={all_R.max().item():.6f}")

        return train_mean, global_mean_logit_score

    def get_logits_score(self, logits):
        if self.logit_method == "MaxLogit":
            return maxLogits(logits)
        elif self.logit_method == "GEN":
            return gen(logits)
        elif self.logit_method == "Energy":
            return energy(logits)
        elif self.logit_method == "MSP":
            return msp(logits)

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []
        w = self.linear.weight.data

        for images, _ in tqdm(data_loader):
            images = images.to(self.device)
            logit, feature = self.model.get_feature(images)

            class_ids = torch.argmax(torch.softmax(logit, dim=1), dim=1).cpu().numpy()
            tm = self.train_mean[class_ids].to(self.device)

            dist = feature - tm
            sg = w[class_ids].sign()

            ep_dist = dist * sg
            ep_dist[ep_dist < 0] = 0
            ep_error = ep_dist.norm(dim=1, p=1) / feature.norm(dim=1, p=1).clamp_min(EPS)

            en_dist = dist * (-sg)
            en_dist[en_dist < 0] = 0
            en_error = en_dist.norm(dim=1, p=1) / feature.norm(dim=1, p=1).clamp_min(EPS)

            logit_score = self.get_logits_score(logit)
            score = ep_error / logit_score + en_error / self.global_mean_logit_score

            result.append(-score.cpu().numpy())

        return np.concatenate(result)


def maxLogits(output):
    return output.max(dim=1).values


def gen(output, gamma=0.1):
    M = output.shape[-1] // 10
    M = 10 if M < 10 else M
    smax = F.softmax(output, dim=1)
    probs_sorted = torch.sort(smax, dim=1).values[:, -M:]
    scores = torch.sum(probs_sorted ** gamma * (1 - probs_sorted) ** gamma, axis=1)
    return 1 / scores


def energy(output):
    return torch.logsumexp(output, dim=1)


def msp(output):
    return torch.max(F.softmax(output, dim=1), dim=1).values