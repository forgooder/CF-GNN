import logging
import os
import statistics

import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from causaleffect import joint_uncond
from utils.utils_cignn import calculate_conditional_MI, calculate_MI


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator):
        self.params = params
        self.graph_classifier = graph_classifier
        self.train_data = train
        self.valid_evaluator = valid_evaluator
        self.updates_counter = 0

        self.optimizer = optim.Adam(graph_classifier.parameters(), lr=params.lr, weight_decay=params.l2)
        self.criterion = nn.MarginRankingLoss(margin=params.margin, reduction='sum')
        self.vae_criterion = nn.BCEWithLogitsLoss(reduction='mean')

        self.train_loader = DataLoader(self.train_data, batch_size=params.batch_size,
                                       shuffle=True, num_workers=params.num_workers,
                                       collate_fn=params.collate_fn)

        self.best_metric = float('inf')
        self.best_auc = 0.0
        self._baseline_debug_logged = False
        self._grad_debug_logged_for_epoch = False

    def train(self):
        self.reset_training_state()
        aux_enabled = self._auxiliary_enabled()
        pretrain_vae_only = getattr(self.params, 'pretrain_vae_only', False)

        if pretrain_vae_only and not getattr(self.params, 'use_vae_loss', False):
            raise ValueError('pretrain_vae_only=true requires use_vae_loss=true.')

        logging.info(
            "Training modes: use_cignn_mask=%s, cignn_mask_mode=%s, "
            "use_vae_loss=%s, use_mi_loss=%s, use_cmi_loss=%s, pretrain_vae_only=%s",
            getattr(self.params, 'use_cignn_mask', False),
            getattr(self.params, 'cignn_mask_mode', 'none'),
            getattr(self.params, 'use_vae_loss', False),
            getattr(self.params, 'use_mi_loss', False),
            getattr(self.params, 'use_cmi_loss', False),
            getattr(self.params, 'pretrain_vae_only', False),
        )
        logging.info("Total loss components: %s", ", ".join(self._loss_component_names()))
        if self._baseline_mode():
            logging.info("BASELINE MODE: CIGNN mask disabled; GraIL message passing is unmodified.")
        elif not getattr(self.params, 'use_cignn_mask', False):
            logging.info("CIGNN mask disabled; auxiliary losses may still be active.")

        for epoch in range(1, self.params.num_epochs + 1):
            logging.info(f'Starting Epoch {epoch}')
            logging.info(
                "Epoch mode: use_cignn_mask=%s, cignn_mask_mode=%s, "
                "use_vae_loss=%s, use_mi_loss=%s, use_cmi_loss=%s, components=%s",
                getattr(self.params, 'use_cignn_mask', False),
                getattr(self.params, 'cignn_mask_mode', 'none'),
                getattr(self.params, 'use_vae_loss', False),
                getattr(self.params, 'use_mi_loss', False),
                getattr(self.params, 'use_cmi_loss', False),
                ",".join(self._loss_component_names()),
            )

            warmup_active = self._warmup_active(epoch)
            current_gamma = self._current_gamma(epoch)
            self.params.current_mask_gamma = current_gamma
            self._grad_debug_logged_for_epoch = False

            if pretrain_vae_only:
                self.set_grads(model_part='vae_mask', requires_grad=True)
                self.set_grads(model_part='classifier', requires_grad=False)
                logging.info(">>> VAE-only pretraining; GraIL predictor is frozen.")
            elif warmup_active:
                self.set_grads(model_part='vae_mask', requires_grad=True)
                self.set_grads(model_part='classifier', requires_grad=False)
                logging.info(">>> Stage 1: Training VAE Decoupling (Encoder/Decoder)")
            else:
                self.set_grads(
                    model_part='vae_mask',
                    requires_grad=aux_enabled and not getattr(self.params, 'freeze_vae_after_pretrain', False)
                )
                self.set_grads(model_part='classifier', requires_grad=True)
                logging.info(">>> Training Link Prediction (GraIL predictor)")

            logging.info("Epoch %s gamma=%s", epoch, current_gamma)

            loss_epoch = []
            self.graph_classifier.train()

            with tqdm(self.train_loader, unit="batch") as tepoch:
                for batch in tepoch:
                    batch = self.params.move_batch_to_device(batch, self.params.device)
                    metrics = self.train_step(batch, epoch)

                    self.updates_counter += 1
                    loss_epoch.append(metrics['total'])
                    tepoch.set_postfix(
                        L_tot=f"{metrics['total']:.3f}",
                        Task=f"{metrics['task']:.3f}",
                        MI=f"{metrics['mi']:.4f}",
                        Mask=f"{metrics['aedge']:.3f}",
                        Adj=f"{metrics['recon_adj']:.4f}"
                    )

            avg_loss = statistics.mean(loss_epoch)
            logging.info(f'Epoch {epoch} Average Loss: {avg_loss}')

            if pretrain_vae_only:
                if avg_loss < self.best_metric:
                    self.best_metric = avg_loss
                    self.save_vae_pretrain(epoch, avg_loss)
                    logging.info(f"VAE pretrain best loss updated: {avg_loss:.4f}, vae_pretrain.pth saved.")
                continue

            if epoch % self.params.eval_every == 0:
                self.graph_classifier.eval()
                result = self.valid_evaluator.eval()
                current_auc = result['auc']
                logging.info(f"Validation result: AUC={current_auc:.4f}, AUC-PR={result['auc_pr']:.4f}")

                if self._warmup_active(epoch):
                    logging.info("Warmup epoch: not saving best_model.pth.")
                else:
                    if current_auc > self.best_auc:
                        self.best_auc = current_auc
                        self.save_classifier(epoch, current_auc)
                        logging.info(f"Stage 2 best AUC updated: {current_auc:.4f}, model saved.")

    def set_grads(self, model_part, requires_grad):
        if model_part == 'vae_mask':
            for p in self.graph_classifier.gnn.encoder.parameters():
                p.requires_grad = requires_grad
            if hasattr(self.graph_classifier.gnn, 'casual_decoder'):
                for p in self.graph_classifier.gnn.casual_decoder.parameters():
                    p.requires_grad = requires_grad
            return

        if model_part == 'classifier':
            for p in self.graph_classifier.fc_layer.parameters():
                p.requires_grad = requires_grad
            for p in self.graph_classifier.rel_emb.parameters():
                p.requires_grad = requires_grad
            for p in self.graph_classifier.gnn.layers.parameters():
                p.requires_grad = requires_grad
            if self.graph_classifier.gnn.attn_rel_emb is not None:
                for p in self.graph_classifier.gnn.attn_rel_emb.parameters():
                    p.requires_grad = requires_grad

    def train_step(self, batch, epoch):
        self.optimizer.zero_grad()

        data_pos, _, data_neg, _ = batch
        g_pos, rel_pos = data_pos
        g_neg, rel_neg = data_neg

        if getattr(self.params, 'pretrain_vae_only', False):
            return self._train_step_vae_pretrain(g_pos, rel_pos, g_neg, rel_neg)

        if not self._auxiliary_enabled():
            pos_scores = self.graph_classifier(data=g_pos, rel_labels=rel_pos).view(-1)
            neg_scores = self.graph_classifier(data=g_neg, rel_labels=rel_neg).view(-1)

            self._assert_baseline_unmodified(g_pos)
            self._assert_baseline_unmodified(g_neg)
            self._log_baseline_debug_once(g_pos, g_neg, 'supervised_loss only')

            loss_task = self._compute_task_loss(pos_scores, neg_scores)
            total_loss = loss_task

            total_loss.backward()
            nn.utils.clip_grad_norm_(self.graph_classifier.parameters(), self.params.clip)
            self.optimizer.step()

            return {
                'total': total_loss.item(),
                'mi': 0.0,
                'aedge': 0.0,
                'recon_adj': 0.0,
                'task': loss_task.item(),
                'components': 'supervised_loss only'
            }

        if not getattr(self.params, 'use_cignn_mask', False) and getattr(self.params, 'cignn_mask_mode', 'none') != 'none':
            raise RuntimeError('use_cignn_mask=false but cignn_mask_mode is not none; refusing to enter mask path.')

        pos_pack = self._encode_graph(g_pos, rel_pos)
        neg_pack = self._encode_graph(g_neg, rel_neg)

        zero = torch.tensor(0.0, device=self.params.device)
        loss_recon_x = zero
        loss_recon_adj = zero
        kld_loss = zero
        vae_loss = zero
        if getattr(self.params, 'use_vae_loss', False):
            loss_recon_x, loss_recon_adj, kld_loss = self._compute_vae_loss(
                pos_pack['g'], pos_pack['node_feats'], pos_pack['z'], pos_pack['mu'], pos_pack['logstd']
            )
            vae_loss = loss_recon_x + loss_recon_adj + kld_loss

        use_mask = (
            getattr(self.params, 'use_cignn_mask', False)
            and getattr(self.params, 'cignn_mask_mode', 'none') != 'none'
        )
        if use_mask:
            causal_pos, pos_scores = joint_uncond(
                pos_pack['alpha'], pos_pack['beta'], pos_pack['g'], pos_pack['rel_labels'],
                self.graph_classifier.gnn.casual_decoder,
                self.graph_classifier,
                self.params.device,
                k=epoch,
                compute_cmi=False
            )
            causal_neg, neg_scores = joint_uncond(
                neg_pack['alpha'], neg_pack['beta'], neg_pack['g'], neg_pack['rel_labels'],
                self.graph_classifier.gnn.casual_decoder,
                self.graph_classifier,
                self.params.device,
                k=epoch,
                compute_cmi=False
            )
            causal_val = 0.5 * (causal_pos + causal_neg)
        else:
            pos_scores = self.graph_classifier(data=pos_pack['g'], rel_labels=pos_pack['rel_labels'])
            neg_scores = self.graph_classifier(data=neg_pack['g'], rel_labels=neg_pack['rel_labels'])
            causal_val = zero

        pos_scores = pos_scores.view(-1)
        neg_scores = neg_scores.view(-1)
        loss_task = self._compute_task_loss(pos_scores, neg_scores)

        loss_mi = zero
        if getattr(self.params, 'use_mi_loss', False):
            alpha_all = torch.cat([pos_pack['alpha'], neg_pack['alpha']], dim=0)
            beta_all = torch.cat([pos_pack['beta'], neg_pack['beta']], dim=0)
            loss_mi = calculate_MI(alpha_all, beta_all)

        loss_cmi = zero
        if getattr(self.params, 'use_cmi_loss', False):
            graph_alpha = torch.cat([pos_pack['graph_alpha'], neg_pack['graph_alpha']], dim=0)
            graph_beta = torch.cat([pos_pack['graph_beta'], neg_pack['graph_beta']], dim=0)
            y_float = torch.cat([
                torch.ones(pos_pack['graph_alpha'].size(0), 1, device=self.params.device),
                torch.zeros(neg_pack['graph_alpha'].size(0), 1, device=self.params.device)
            ], dim=0)
            loss_cmi = calculate_conditional_MI(graph_alpha, y_float, graph_beta)

        if self._warmup_active(epoch):
            total_loss = 0.1 * loss_task
            if getattr(self.params, 'use_vae_loss', False):
                total_loss = total_loss + self.params.lambda_vae * vae_loss
            if getattr(self.params, 'use_mi_loss', False):
                total_loss = total_loss + self.params.lambda_mi * loss_mi
            if getattr(self.params, 'use_cmi_loss', False):
                total_loss = total_loss - self.params.lambda_cmi * loss_cmi
        else:
            total_loss = loss_task
            if getattr(self.params, 'use_vae_loss', False):
                total_loss = total_loss + 0.1 * vae_loss
            if getattr(self.params, 'use_mi_loss', False):
                total_loss = total_loss + self.params.lambda_mi * loss_mi
            if getattr(self.params, 'use_cmi_loss', False):
                total_loss = total_loss - self.params.lambda_cmi * loss_cmi

        mask_log = self._log_mask_stats(pos_pack['g'], neg_pack['g']) if use_mask else None
        aedge_mean = mask_log['raw']['mean'] if mask_log else 0.0

        total_loss.backward()
        self._log_grad_debug_if_needed(
            epoch=epoch,
            g_pos=pos_pack['g'],
            g_neg=neg_pack['g'],
            prediction_loss=loss_task,
            total_loss=total_loss,
        )
        nn.utils.clip_grad_norm_(self.graph_classifier.parameters(), self.params.clip)
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'mi': loss_mi.item(),
            'aedge': aedge_mean,
            'recon_adj': loss_recon_adj.item(),
            'task': loss_task.item(),
            'components': ",".join(self._loss_component_names())
        }

    def _train_step_vae_pretrain(self, g_pos, rel_pos, g_neg, rel_neg):
        pos_pack = self._encode_graph(g_pos, rel_pos)
        neg_pack = self._encode_graph(g_neg, rel_neg)

        pos_recon_x, pos_recon_adj, pos_kld = self._compute_vae_loss(
            pos_pack['g'], pos_pack['node_feats'], pos_pack['z'], pos_pack['mu'], pos_pack['logstd']
        )
        neg_recon_x, neg_recon_adj, neg_kld = self._compute_vae_loss(
            neg_pack['g'], neg_pack['node_feats'], neg_pack['z'], neg_pack['mu'], neg_pack['logstd']
        )
        loss_recon_x = 0.5 * (pos_recon_x + neg_recon_x)
        loss_recon_adj = 0.5 * (pos_recon_adj + neg_recon_adj)
        kld_loss = 0.5 * (pos_kld + neg_kld)
        vae_loss = loss_recon_x + loss_recon_adj + kld_loss

        zero = torch.tensor(0.0, device=self.params.device)
        loss_mi = zero
        if getattr(self.params, 'use_mi_loss', False):
            alpha_all = torch.cat([pos_pack['alpha'], neg_pack['alpha']], dim=0)
            beta_all = torch.cat([pos_pack['beta'], neg_pack['beta']], dim=0)
            loss_mi = calculate_MI(alpha_all, beta_all)

        loss_cmi = zero
        if getattr(self.params, 'use_cmi_loss', False):
            graph_alpha = torch.cat([pos_pack['graph_alpha'], neg_pack['graph_alpha']], dim=0)
            graph_beta = torch.cat([pos_pack['graph_beta'], neg_pack['graph_beta']], dim=0)
            y_float = torch.cat([
                torch.ones(pos_pack['graph_alpha'].size(0), 1, device=self.params.device),
                torch.zeros(neg_pack['graph_alpha'].size(0), 1, device=self.params.device)
            ], dim=0)
            loss_cmi = calculate_conditional_MI(graph_alpha, y_float, graph_beta)

        total_loss = self.params.lambda_vae * vae_loss
        if getattr(self.params, 'use_mi_loss', False):
            total_loss = total_loss + self.params.lambda_mi * loss_mi
        if getattr(self.params, 'use_cmi_loss', False):
            total_loss = total_loss - self.params.lambda_cmi * loss_cmi

        mask = torch.cat([
            self._compute_raw_mask(pos_pack['g'], pos_pack['alpha']),
            self._compute_raw_mask(neg_pack['g'], neg_pack['alpha'])
        ], dim=0)
        mask_stats = self._mask_stats(mask)

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.graph_classifier.parameters(), self.params.clip)
        self.optimizer.step()

        logging.info(
            "VAE pretrain batch: total=%.6f recon_x=%.6f recon_adj=%.6f kl=%.6f "
            "mi=%.6f cmi=%.6f mask_mean=%.6f mask_min=%.6f mask_max=%.6f "
            "mask_std=%.6f pct_lt_0.2=%.6f pct_gt_0.8=%.6f",
            total_loss.item(),
            loss_recon_x.item(),
            loss_recon_adj.item(),
            kld_loss.item(),
            loss_mi.item(),
            loss_cmi.item(),
            mask_stats['mean'],
            mask_stats['min'],
            mask_stats['max'],
            mask_stats['std'],
            mask_stats['pct_lt_0_2'],
            mask_stats['pct_gt_0_8'],
        )

        return {
            'total': total_loss.item(),
            'mi': loss_mi.item(),
            'aedge': mask_stats['mean'],
            'recon_adj': loss_recon_adj.item(),
            'task': 0.0,
            'components': ",".join(self._loss_component_names()),
            'vae_total': vae_loss.item(),
            'recon_x': loss_recon_x.item(),
            'kl': kld_loss.item(),
            'mask_min': mask_stats['min'],
            'mask_max': mask_stats['max'],
            'mask_std': mask_stats['std'],
            'mask_pct_lt_0_2': mask_stats['pct_lt_0_2'],
            'mask_pct_gt_0_8': mask_stats['pct_gt_0_8'],
        }

    def _compute_task_loss(self, pos_scores, neg_scores):
        num_neg = self.params.num_neg_samples_per_link
        expected_neg = pos_scores.numel() * num_neg
        if neg_scores.numel() != expected_neg:
            raise ValueError(
                f"Negative score count mismatch: got {neg_scores.numel()}, expected {expected_neg}. "
                "Check num_neg_samples_per_link and collate_dgl."
            )

        pos_scores_for_loss = pos_scores.repeat_interleave(num_neg, dim=0)
        target = torch.ones_like(neg_scores, device=self.params.device)
        return self.criterion(pos_scores_for_loss, neg_scores, target)

    def _auxiliary_enabled(self):
        return (
            getattr(self.params, 'use_cignn_mask', False)
            or getattr(self.params, 'use_vae_loss', False)
            or getattr(self.params, 'use_mi_loss', False)
            or getattr(self.params, 'use_cmi_loss', False)
            or getattr(self.params, 'pretrain_vae_only', False)
        )

    def _baseline_mode(self):
        return not self._auxiliary_enabled()

    def _warmup_active(self, epoch):
        if getattr(self.params, 'pretrain_vae_only', False):
            return False
        if not self._auxiliary_enabled():
            return False
        if getattr(self.params, 'load_pretrained_vae', ''):
            return False
        if getattr(self.params, 'freeze_vae_after_pretrain', False):
            return False
        return epoch <= getattr(self.params, 'warmup_epochs', 50)

    def _current_gamma(self, epoch):
        target_gamma = getattr(self.params, 'mask_injection_gamma', 0.0)
        if getattr(self.params, 'pretrain_vae_only', False):
            return 0.0
        if self._warmup_active(epoch):
            return 0.0

        schedule = getattr(self.params, 'mask_gamma_schedule', 'none')
        if schedule == 'none':
            return target_gamma
        if schedule != 'linear':
            raise ValueError(f"Unsupported mask_gamma_schedule: {schedule}")

        ramp_epochs = getattr(self.params, 'mask_ramp_epochs', 0)
        if ramp_epochs <= 0:
            return target_gamma

        warmup_epochs = 0
        if (
            self._auxiliary_enabled()
            and not getattr(self.params, 'load_pretrained_vae', '')
            and not getattr(self.params, 'freeze_vae_after_pretrain', False)
        ):
            warmup_epochs = getattr(self.params, 'warmup_epochs', 50)
        joint_epoch = max(epoch - warmup_epochs, 0)
        if ramp_epochs == 1:
            return target_gamma
        scale = min(max(float(joint_epoch - 1), 0.0) / float(ramp_epochs - 1), 1.0)
        return target_gamma * scale

    def _loss_component_names(self):
        if self._baseline_mode():
            return ['supervised_loss only']
        components = []
        if not getattr(self.params, 'pretrain_vae_only', False):
            components.append('task')
        if getattr(self.params, 'use_vae_loss', False):
            components.append('vae')
        if getattr(self.params, 'use_mi_loss', False):
            components.append('mi')
        if getattr(self.params, 'use_cmi_loss', False):
            components.append('cmi')
        if (
            getattr(self.params, 'use_cignn_mask', False)
            and getattr(self.params, 'cignn_mask_mode', 'none') != 'none'
        ):
            components.append('causal_mask_effect')
        return components

    def _compute_raw_mask(self, g, alpha):
        u, v = g.edges()
        z_causal = self.graph_classifier.gnn.casual_decoder(alpha)
        mask_logits = torch.sum(z_causal[u] * z_causal[v], dim=1)
        return torch.sigmoid(mask_logits).view(-1, 1)

    def _tensor_stats(self, tensor):
        tensor_flat = tensor.detach().view(-1)
        if tensor_flat.numel() == 0:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0}
        return {
            'mean': tensor_flat.mean().item(),
            'min': tensor_flat.min().item(),
            'max': tensor_flat.max().item(),
            'std': tensor_flat.std(unbiased=False).item(),
        }

    def _collect_edge_tensor(self, graphs, key):
        tensors = [g.edata[key] for g in graphs if key in g.edata]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    def _log_mask_stats(self, g_pos, g_neg):
        raw_mask = self._collect_edge_tensor([g_pos, g_neg], 'raw_cignn_mask')
        effective_mask = self._collect_edge_tensor([g_pos, g_neg], 'effective_cignn_mask')
        if raw_mask is None or effective_mask is None:
            raise RuntimeError('CIGNN mask was enabled but raw/effective mask tensors were not generated.')

        raw_stats = self._tensor_stats(raw_mask)
        effective_stats = self._tensor_stats(effective_mask)
        message_applied = (
            getattr(g_pos, 'message_mask_applied', False)
            or getattr(g_neg, 'message_mask_applied', False)
        )
        attention_applied = (
            getattr(g_pos, 'attention_mask_applied', False)
            or getattr(g_neg, 'attention_mask_applied', False)
        )
        logging.info(
            "Mask stats: use_cignn_mask=%s mode=%s gamma=%.6f "
            "raw_mean=%.6f raw_min=%.6f raw_max=%.6f raw_std=%.6f "
            "effective_mean=%.6f effective_min=%.6f effective_max=%.6f effective_std=%.6f "
            "message_mask_applied=%s attention_mask_applied=%s",
            getattr(self.params, 'use_cignn_mask', False),
            getattr(self.params, 'cignn_mask_mode', 'none'),
            getattr(self.params, 'current_mask_gamma', 0.0),
            raw_stats['mean'],
            raw_stats['min'],
            raw_stats['max'],
            raw_stats['std'],
            effective_stats['mean'],
            effective_stats['min'],
            effective_stats['max'],
            effective_stats['std'],
            message_applied,
            attention_applied,
        )
        return {'raw': raw_stats, 'effective': effective_stats}

    def _grad_norm(self, parameters):
        total = 0.0
        has_grad = False
        for parameter in parameters:
            if parameter.grad is None:
                continue
            has_grad = True
            total += parameter.grad.detach().pow(2).sum().item()
        if not has_grad:
            return None
        return total ** 0.5

    def _log_grad_debug_if_needed(self, epoch, g_pos, g_neg, prediction_loss, total_loss):
        if not getattr(self.params, 'debug_grad_check', False):
            return
        if self._grad_debug_logged_for_epoch:
            return

        predictor_params = list(self.graph_classifier.fc_layer.parameters())
        predictor_params += list(self.graph_classifier.rel_emb.parameters())
        predictor_params += list(self.graph_classifier.gnn.layers.parameters())
        if self.graph_classifier.gnn.attn_rel_emb is not None:
            predictor_params += list(self.graph_classifier.gnn.attn_rel_emb.parameters())

        encoder_grad_norm = self._grad_norm(self.graph_classifier.gnn.encoder.parameters())
        decoder_grad_norm = self._grad_norm(self.graph_classifier.gnn.casual_decoder.parameters())
        predictor_grad_norm = self._grad_norm(predictor_params)

        mask_logits = getattr(g_pos, 'mask_logits_ref', None)
        raw_mask = getattr(g_pos, 'raw_mask_ref', None)
        effective_mask = getattr(g_pos, 'effective_mask_ref', None)
        gamma = getattr(self.params, 'current_mask_gamma', 0.0)

        logging.info("Gradient sanity check epoch %s:", epoch)
        logging.info("  GraIL predictor grad norm: %s", predictor_grad_norm)
        logging.info("  VAE/mask encoder grad norm: %s", encoder_grad_norm)
        logging.info("  VAE/mask decoder grad norm: %s", decoder_grad_norm)
        logging.info("  mask_logits.requires_grad: %s", bool(mask_logits.requires_grad) if mask_logits is not None else False)
        logging.info("  raw_mask.requires_grad: %s", bool(raw_mask.requires_grad) if raw_mask is not None else False)
        logging.info("  effective_mask.requires_grad: %s", bool(effective_mask.requires_grad) if effective_mask is not None else False)
        logging.info("  effective_mask.grad_fn: %s", effective_mask.grad_fn if effective_mask is not None else None)
        logging.info("  prediction_loss.requires_grad: %s", bool(prediction_loss.requires_grad))
        logging.info("  total_loss.requires_grad: %s", bool(total_loss.requires_grad))
        logging.info("  loss components: %s", ",".join(self._loss_component_names()))

        if getattr(self.params, 'freeze_vae_after_pretrain', False):
            logging.info("VAE is frozen; no VAE gradients expected.")
        elif not getattr(self.params, 'use_cignn_mask', False):
            logging.info("CIGNN mask disabled; VAE/mask branch is not in GraIL message passing.")
        elif gamma > 0:
            encoder_missing = encoder_grad_norm is None or encoder_grad_norm == 0.0
            decoder_missing = decoder_grad_norm is None or decoder_grad_norm == 0.0
            if encoder_missing or decoder_missing:
                logging.warning("WARNING: VAE/mask branch receives no gradient from prediction loss.")

        self._grad_debug_logged_for_epoch = True

    def _mask_stats(self, mask):
        mask_flat = mask.detach().view(-1)
        if mask_flat.numel() == 0:
            return {
                'mean': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0,
                'pct_lt_0_2': 0.0, 'pct_gt_0_8': 0.0
            }
        return {
            'mean': mask_flat.mean().item(),
            'min': mask_flat.min().item(),
            'max': mask_flat.max().item(),
            'std': mask_flat.std(unbiased=False).item(),
            'pct_lt_0_2': (mask_flat < 0.2).float().mean().item(),
            'pct_gt_0_8': (mask_flat > 0.8).float().mean().item(),
        }

    def _assert_baseline_unmodified(self, g):
        if 'causal_mask' in g.edata:
            raise RuntimeError('Baseline mode generated causal_mask; GraIL path is not unmodified.')
        if 'raw_cignn_mask' in g.edata or 'effective_cignn_mask' in g.edata:
            raise RuntimeError('Baseline mode generated CIGNN mask tensors; GraIL path is not unmodified.')
        if getattr(g, 'vae_encoder_called', False):
            raise RuntimeError('Baseline mode called VAE encoder; GraIL path is not unmodified.')
        if getattr(g, 'message_mask_applied', False):
            raise RuntimeError('Baseline mode applied message mask; GraIL path is not unmodified.')
        if getattr(g, 'attention_mask_applied', False):
            raise RuntimeError('Baseline mode applied attention mask; GraIL path is not unmodified.')

    def _log_baseline_debug_once(self, g_pos, g_neg, components):
        if not getattr(self.params, 'debug_baseline_check', False):
            return
        if self._baseline_debug_logged:
            return

        causal_mask_is_none = 'causal_mask' not in g_pos.edata and 'causal_mask' not in g_neg.edata
        vae_encoder_called = (
            getattr(g_pos, 'vae_encoder_called', False)
            or getattr(g_neg, 'vae_encoder_called', False)
        )
        message_mask_applied = (
            getattr(g_pos, 'message_mask_applied', False)
            or getattr(g_neg, 'message_mask_applied', False)
        )
        attention_mask_applied = (
            getattr(g_pos, 'attention_mask_applied', False)
            or getattr(g_neg, 'attention_mask_applied', False)
        )
        logging.info("Baseline debug check:")
        logging.info("  use_cignn_mask: %s", getattr(self.params, 'use_cignn_mask', False))
        logging.info("  cignn_mask_mode: %s", getattr(self.params, 'cignn_mask_mode', 'none'))
        logging.info("  use_vae_loss: %s", getattr(self.params, 'use_vae_loss', False))
        logging.info("  use_mi_loss: %s", getattr(self.params, 'use_mi_loss', False))
        logging.info("  use_cmi_loss: %s", getattr(self.params, 'use_cmi_loss', False))
        logging.info("  pretrain_vae_only: %s", getattr(self.params, 'pretrain_vae_only', False))
        logging.info("  causal_mask is None: %s", causal_mask_is_none)
        logging.info("  loss components: %s", components)
        logging.info("  VAE encoder called: %s", vae_encoder_called)
        logging.info("  message mask applied: %s", message_mask_applied)
        logging.info("  attention mask applied: %s", attention_mask_applied)
        self._baseline_debug_logged = True

    def _encode_graph(self, g, rel_labels):
        g = g.to(self.params.device)
        rel_labels = rel_labels.to(self.params.device)
        node_feats = g.ndata['feat']

        if 'id' not in g.ndata:
            g.ndata['id'] = torch.zeros(g.num_nodes(), device=self.params.device)
            g.ndata['id'][(node_feats[:, 0] == 1)] = 1
            g.ndata['id'][(node_feats[:, 1] == 1)] = 2

        z, mu, logstd = self.graph_classifier.gnn.encoder(g, node_feats)
        mid = z.shape[1] // 2
        alpha = z[:, :mid]
        beta = z[:, mid:]

        g.ndata['alpha_tmp'] = alpha
        g.ndata['beta_tmp'] = beta
        graph_alpha = dgl.mean_nodes(g, 'alpha_tmp')
        graph_beta = dgl.mean_nodes(g, 'beta_tmp')
        g.ndata.pop('alpha_tmp')
        g.ndata.pop('beta_tmp')

        return {
            'g': g,
            'rel_labels': rel_labels,
            'node_feats': node_feats,
            'z': z,
            'mu': mu,
            'logstd': logstd,
            'alpha': alpha,
            'beta': beta,
            'graph_alpha': graph_alpha,
            'graph_beta': graph_beta
        }

    def _compute_vae_loss(self, g, node_feats, z, mu, logstd):
        x_rec = self.graph_classifier.gnn.encoder.reconstruct_decoder(z)
        loss_recon_x = nn.functional.mse_loss(x_rec, node_feats)

        u, v = g.edges()
        pos_logits = torch.sum(z[u] * z[v], dim=1)
        num_edges = pos_logits.numel()
        neg_u = torch.randint(0, g.num_nodes(), (num_edges,), device=self.params.device)
        neg_v = torch.randint(0, g.num_nodes(), (num_edges,), device=self.params.device)
        neg_logits = torch.sum(z[neg_u] * z[neg_v], dim=1)
        recon_logits = torch.cat([pos_logits, neg_logits], dim=0)
        recon_labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits)
        ], dim=0)
        loss_recon_adj = self.vae_criterion(recon_logits, recon_labels)

        kld_loss = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1))
        return loss_recon_x, loss_recon_adj, kld_loss

    def reset_training_state(self):
        self.updates_counter = 0
        self._baseline_debug_logged = False
        self._grad_debug_logged_for_epoch = False

    def _checkpoint_params(self):
        serializable = {}
        for key, value in vars(self.params).items():
            if key in {'collate_fn', 'move_batch_to_device'}:
                continue
            try:
                import json
                json.dumps(value)
                serializable[key] = value
            except TypeError:
                serializable[key] = str(value)
        return serializable

    def _base_checkpoint(self, epoch, metric, is_full_prediction_model, checkpoint_type):
        return {
            'model_state_dict': self.graph_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_metric': metric,
            'params': self._checkpoint_params(),
            'use_cignn_mask': getattr(self.params, 'use_cignn_mask', False),
            'cignn_mask_mode': getattr(self.params, 'cignn_mask_mode', 'none'),
            'gamma': getattr(self.params, 'current_mask_gamma', getattr(self.params, 'mask_injection_gamma', 0.0)),
            'is_full_prediction_model': is_full_prediction_model,
            'checkpoint_type': checkpoint_type,
        }

    def save_classifier(self, epoch=None, metric=None):
        checkpoint_path = os.path.join(self.params.exp_dir, 'best_model.pth')
        checkpoint = self._base_checkpoint(
            epoch=epoch,
            metric=metric,
            is_full_prediction_model=True,
            checkpoint_type='full_prediction',
        )
        torch.save(checkpoint, checkpoint_path)

    def save_vae_pretrain(self, epoch=None, metric=None):
        checkpoint_path = os.path.join(self.params.exp_dir, 'vae_pretrain.pth')
        branch_prefixes = ('gnn.encoder.', 'gnn.casual_decoder.')
        branch_state = {
            key: value for key, value in self.graph_classifier.state_dict().items()
            if key.startswith(branch_prefixes)
        }
        checkpoint = self._base_checkpoint(
            epoch=epoch,
            metric=metric,
            is_full_prediction_model=False,
            checkpoint_type='vae_pretrain',
        )
        checkpoint['model_state_dict'] = branch_state
        torch.save(checkpoint, checkpoint_path)
