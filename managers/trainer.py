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

    def train(self):
        self.reset_training_state()
        warmup_epochs = getattr(self.params, 'warmup_epochs', 50)

        for epoch in range(1, self.params.num_epochs + 1):
            logging.info(f'Starting Epoch {epoch}')

            if epoch <= warmup_epochs:
                self.set_grads(model_part='vae', requires_grad=True)
                self.set_grads(model_part='classifier', requires_grad=False)
                logging.info(">>> Stage 1: Training VAE Decoupling (Encoder/Decoder)")
            else:
                self.set_grads(model_part='vae', requires_grad=False)
                self.set_grads(model_part='classifier', requires_grad=True)
                logging.info(">>> Stage 2: Training Link Prediction (Classifier)")

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

            if epoch % self.params.eval_every == 0:
                self.graph_classifier.eval()
                result = self.valid_evaluator.eval()
                current_auc = result['auc']
                logging.info(f"Validation result: AUC={current_auc:.4f}, AUC-PR={result['auc_pr']:.4f}")

                if epoch <= warmup_epochs:
                    if avg_loss < self.best_metric:
                        self.best_metric = avg_loss
                        self.save_classifier()
                        logging.info(f"Stage 1 best loss updated: {avg_loss:.4f}, model saved.")
                else:
                    if current_auc > self.best_auc:
                        self.best_auc = current_auc
                        self.save_classifier()
                        logging.info(f"Stage 2 best AUC updated: {current_auc:.4f}, model saved.")

    def set_grads(self, model_part, requires_grad):
        if model_part == 'vae':
            for p in self.graph_classifier.gnn.encoder.parameters():
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
            if hasattr(self.graph_classifier.gnn, 'casual_decoder'):
                for p in self.graph_classifier.gnn.casual_decoder.parameters():
                    p.requires_grad = requires_grad

    def train_step(self, batch, epoch):
        self.optimizer.zero_grad()

        data_pos, _, data_neg, _ = batch
        g_pos, rel_pos = data_pos
        g_neg, rel_neg = data_neg

        pos_pack = self._encode_graph(g_pos, rel_pos)
        neg_pack = self._encode_graph(g_neg, rel_neg)

        loss_recon_x, loss_recon_adj, kld_loss = self._compute_vae_loss(
            pos_pack['g'], pos_pack['node_feats'], pos_pack['z'], pos_pack['mu'], pos_pack['logstd']
        )
        vae_loss = loss_recon_x + loss_recon_adj + kld_loss

        causal_pos, pos_scores = joint_uncond(
            pos_pack['alpha'], pos_pack['beta'], pos_pack['g'], pos_pack['rel_labels'],
            self.graph_classifier.gnn.casual_decoder,
            self.graph_classifier,
            self.params.device,
            k=epoch
        )
        causal_neg, neg_scores = joint_uncond(
            neg_pack['alpha'], neg_pack['beta'], neg_pack['g'], neg_pack['rel_labels'],
            self.graph_classifier.gnn.casual_decoder,
            self.graph_classifier,
            self.params.device,
            k=epoch
        )
        causal_val = 0.5 * (causal_pos + causal_neg)

        pos_scores = pos_scores.view(-1)
        neg_scores = neg_scores.view(-1)
        num_neg = self.params.num_neg_samples_per_link
        expected_neg = pos_scores.numel() * num_neg
        if neg_scores.numel() != expected_neg:
            raise ValueError(
                f"Negative score count mismatch: got {neg_scores.numel()}, expected {expected_neg}. "
                "Check num_neg_samples_per_link and collate_dgl."
            )

        pos_scores_for_loss = pos_scores.repeat_interleave(num_neg, dim=0)
        target = torch.ones_like(neg_scores, device=self.params.device)
        loss_task = self.criterion(pos_scores_for_loss, neg_scores, target)

        alpha_all = torch.cat([pos_pack['alpha'], neg_pack['alpha']], dim=0)
        beta_all = torch.cat([pos_pack['beta'], neg_pack['beta']], dim=0)
        loss_mi = calculate_MI(alpha_all, beta_all)

        graph_alpha = torch.cat([pos_pack['graph_alpha'], neg_pack['graph_alpha']], dim=0)
        graph_beta = torch.cat([pos_pack['graph_beta'], neg_pack['graph_beta']], dim=0)
        y_float = torch.cat([
            torch.ones(pos_pack['graph_alpha'].size(0), 1, device=self.params.device),
            torch.zeros(neg_pack['graph_alpha'].size(0), 1, device=self.params.device)
        ], dim=0)
        loss_cmi = calculate_conditional_MI(graph_alpha, y_float, graph_beta)

        warmup_steps = getattr(self.params, 'warmup_epochs', 50)
        if epoch <= warmup_steps:
            total_loss = (self.params.lambda_vae * vae_loss +
                          self.params.lambda_mi * loss_mi -
                          self.params.lambda_cmi * loss_cmi -
                          0.01 * causal_val +
                          0.1 * loss_task)
        else:
            total_loss = (loss_task +
                          0.1 * vae_loss +
                          self.params.lambda_mi * loss_mi -
                          self.params.lambda_cmi * loss_cmi -
                          0.05 * causal_val)

        with torch.no_grad():
            mask_vals = []
            if 'causal_mask' in pos_pack['g'].edata:
                mask_vals.append(pos_pack['g'].edata['causal_mask'].mean())
            if 'causal_mask' in neg_pack['g'].edata:
                mask_vals.append(neg_pack['g'].edata['causal_mask'].mean())
            aedge_mean = torch.stack(mask_vals).mean().item() if mask_vals else 0.0

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.graph_classifier.parameters(), self.params.clip)
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'mi': loss_mi.item(),
            'aedge': aedge_mean,
            'recon_adj': loss_recon_adj.item(),
            'task': loss_task.item()
        }

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

    def save_classifier(self):
        checkpoint_path = os.path.join(self.params.exp_dir, 'best_model.pth')
        torch.save(self.graph_classifier.state_dict(), checkpoint_path)
