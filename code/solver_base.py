import os
import numpy as np

from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import torch.nn.functional as F
from data import MyDataset
from models.multimodal import EarlyFusion
from metr import compute_ccc_batched, compute_r2_score_batched, compute_pearson_correlation_batched
import wandb


class FocalBCELoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure targets are in the correct shape
        #targets = targets.view_as(inputs)  # Reshape targets to match input shape if not already
        BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
        #F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class CCCLoss1(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, x, y):
        ccc = 2*torch.cov(x, y) / (x.var() + y.var() + (x.mean() - y.mean())**2)
        return ccc
class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        xy = torch.stack((x, y), dim=0)
        cov_matrix = torch.cov(xy)
        cov_xy = cov_matrix[0, 1]
        ccc = 2 * cov_xy / (x.var() + y.var() + (x.mean() - y.mean())**2)
        ccc = -ccc# lower is better
        return ccc
class solver_base(nn.Module):
    def __init__(self, config):
        super(solver_base, self).__init__()
        self.config = config

        # Initiate the networks and data loaders
        self.model = EarlyFusion(config).cuda()

        # Load pre-trained weights
        # ckpt_path = TODO
        # ckpt = torch.load(ckpt_path)
        # del ckpt["interpreter.8.weight"]
        # del ckpt["interpreter.8.bias"]
        # self.model.load_state_dict(ckpt, strict=False)

        self.get_data_loaders()

        # Setup the optimizers and loss function
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        if self.config.targettype == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif self.config.targettype == 'mse':
            self.criterion = nn.MSELoss()
        elif self.config.targettype == 'focal':
            self.criterion =FocalBCELoss()#nn.CrossEntropyLoss() # FocalBCELoss()
        elif self.config.targettype == 'ccc':
            self.criterion = CCCLoss()

        if self.config.scheduler == 'linear':
            #linear LinearLR
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
        else: 
            self.scheduler = None


    def get_data_loaders(self):
        dataset = self.config.dataset
        data_root = self.config.data_root
        label_root = self.config.label_root
        batch_size = self.config.batch_size
        num_workers = self.config.num_workers

        train_csv_path = f"{data_root}/{dataset}/{label_root}/train.csv"
        val_csv_path = f"{data_root}/{dataset}/{label_root}/val.csv"
        if self.config.kfolds:
            test_csv_path = val_csv_path
        else:
            test_csv_path = f"{data_root}/{dataset}/{label_root}/test.csv"

        train_set = MyDataset(train_csv_path, True, self.config)
        dev_set = MyDataset(val_csv_path, False, self.config)
        test_set = MyDataset(test_csv_path, False, self.config)

        self.train_loader = DataLoader(
            train_set,
            sampler=ImbalancedDatasetSampler(train_set),
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=train_set.collate_fn,
            # shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(dev_set, batch_size=batch_size, num_workers=num_workers, collate_fn=dev_set.collate_fn, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, collate_fn=test_set.collate_fn, shuffle=False)


    def test_model(self, test_loader, mode = "val"):
        with torch.no_grad():
            self.eval()
            test_loss = []
            gt_list, pred_list = [], []
            for (videos, audio_paths, texts, labels, openfacefeat) in tqdm(test_loader):
                videos = videos.cuda()
                labels = labels.cuda()

                preds = self.model(videos, audio_paths, texts, openfacefeat)
                loss = self.criterion(preds, labels)

                test_loss.append(loss.item())
                if self.config.targettype == 'ce' or self.config.targettype == 'focal':
                    preds = torch.argmax(preds, dim=1)
                ####################import pdb; pdb.set_trace()
                print(preds)
                gt_list.append(labels)
                pred_list.append(preds)
            gt_list = torch.cat(gt_list, dim=0).detach().cpu().numpy()
            pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
            if self.config.targettype == 'ce' or self.config.targettype == 'focal':
                f1 = 100.0 * f1_score(gt_list, pred_list, average="macro")
                acc = 100.0 * balanced_accuracy_score(gt_list, pred_list)
                met = {"f1": f1, "acc": acc, "met" : f1}
            elif self.config.targettype == 'mse' or self.config.targettype == 'ccc':
                ccc = compute_ccc_batched(y_pred = pred_list, y_true = gt_list)
                r2 = compute_r2_score_batched(y_pred = pred_list, y_true = gt_list)
                #mse
                criterionloss = self.criterion(torch.tensor(pred_list), torch.tensor(gt_list)).item()
                mse_loss = np.mean((pred_list - gt_list)**2)
                pcc = compute_pearson_correlation_batched(y_pred = pred_list, y_true = gt_list)
                met = {"ccc": ccc, "r2": r2, "met" : ccc, "criterion": criterionloss, "mse": mse_loss, "pcc": pcc}
            if mode == "test":
                met = {"test_" + k: v for k, v in met.items()}
            elif mode == "val":
                met = {"val_" + k: v for k, v in met.items()}
            elif mode == "train":
                met = {"train_" + k: v for k, v in met.items()}
            wandb.log(met)
            test_loss = [i for i in test_loss if str(i) != 'nan']
            test_loss_avg = sum(test_loss) / len(test_loss)
            #import pdb; pdb.set_trace()
            return test_loss_avg, met


    def run(self):
        best_val_f1 = 0.
        patience = self.config.patience

        for epochs in range(1, self.config.num_epochs+1):
            print(f"Epoch: {epochs}/{self.config.num_epochs}")
            self.train()
            train_loss = []
            for (videos, audio_paths, texts, labels, openfacefeat) in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                videos = videos.cuda()
                labels = labels.cuda()

                preds = self.model(videos, audio_paths, texts, openfacefeat)
                loss = self.criterion(preds, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

                train_loss.append(loss.item())
            if self.scheduler != None:
                self.scheduler.step()

            train_loss_avg = sum(train_loss) / len(train_loss)
            wandb.log({"train_loss": train_loss_avg})

            # Validate model
            if self.config.trainmets:
                _, trainmet = self.test_model(self.train_loader, mode = "train")
                wandb.log(trainmet)
            

            val_loss_avg, met = self.test_model(self.val_loader, mode = "val")
            wandb.log({"val_loss": val_loss_avg})
            log_str = f"Train loss: {train_loss_avg:.4f} Val loss: {val_loss_avg:.4f}"
            for k, v in met.items():
                log_str += f" {k}: {v:.9f}"
            if self.config.trainmets:
                for k, v in trainmet.items():
                    log_str += f" {k}: {v:.9f}"
            print(log_str)
            val_f1 = met["val_met"]

            if val_f1 > best_val_f1:
                patience = self.config.patience
                best_val_f1 = val_f1
                os.makedirs(f"{self.config.ckpt_root}/{self.config.dataset}", exist_ok=True)
                ckpt_path = f"{self.config.ckpt_root}/{self.config.dataset}/{self.config.ckpt_name}.pt"
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"save to: {ckpt_path}")
            else:
                patience -= 1
                if patience == 0:
                    break

        # Test model
        ckpt_path = f"{self.config.ckpt_root}/{self.config.dataset}/{self.config.ckpt_name}.pt"
        self.model.load_state_dict(torch.load(ckpt_path))
        test_loss_avg, met = self.test_model(self.test_loader, mode = "test")
        wandb.log({"test_loss": test_loss_avg})
        log_str = f"Test loss: {test_loss_avg:.4f}"
        for k, v in met.items():
            log_str += f" {k}: {v:.9f}"
    def run_eval(self):
        # Test model
        ckpt_path = f"{self.config.ckpt_root}/{self.config.dataset}/{self.config.ckpt_name}.pt"
        self.model.load_state_dict(torch.load(ckpt_path))
        test_loss_avg, met = self.test_model(self.test_loader, mode = "test")
        import pdb; pdb.set_trace()
        wandb.log({"test_loss": test_loss_avg})
        log_str = f"Test loss: {test_loss_avg:.4f}"
        for k, v in met.items():
            log_str += f" {k}: {v:.9f}"
        print(log_str)