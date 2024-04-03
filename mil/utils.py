from mil.set_transformer import PMA,SAB
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score
import pickle
import math
import sys
from cmcrameri import cm
import matplotlib.font_manager as font_manager
sys.path.insert(0, "../")
from train_diff_autoenc import LitModel

font_dir = '/mnt/bulk/laura/diffae/wanshi-utils/HelveticaNeue.ttf'
my_font = font_manager.FontProperties(fname=font_dir)

class Classifier(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, num_classes):
        super(Classifier, self).__init__()
        self.dim = dim
        self.pool = nn.Sequential(
                    PMA(dim,num_heads,num_seeds),
                    SAB(dim,dim,num_heads),
                    )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(dim,num_classes),
            )                        # empirical evidence showed that a non-linear head can improve performance
        #nn.init.kaiming_normal_(self.classifier.weight)
        
    def forward(self,x):
        x = self.pool(x).max(1).values    # Taking the max here probably makes more sense
        return self.classifier(x)
    
class FeatDataset(Dataset):
    def __init__(self, feat_list, annot_file, target_label, target_dict, nr_feats, indices=None):
        self.feat_list = feat_list
        self.indices = indices if indices is not None else list(range(len(self.feat_list)))
        try:
            if annot_file.endswith(".tsv"):
                self.df = pd.read_csv(annot_file,sep="\t")
            else:    
                self.df = pd.read_csv(annot_file)
        except Exception:
            self.df = pd.read_excel(annot_file)
        self.df['PATIENT'] = self.df['PATIENT'].apply(lambda patient: "-".join(patient.split("-")[:3]))
        self.target_label = target_label
        self.target_dict = target_dict
        self.nr_feats = nr_feats

    def __len__(self):
        return len(self.indices)

    def get_targets(self, indices=None):
        indices = indices if indices is not None else self.indices
        return [
            self.target_dict[self.df[self.df.PATIENT == ("-").join(self.feat_list[i].split("/")[-1].split(".h5")[0].split("-")[:3])][self.target_label].values[0]]
                for i in indices]

    def get_nr_pos(self, indices=None):
        #return len(self.df[self.df[self.target_label]==pos_label].values)
        #targets = np.array([self.target_dict[self.df[self.df.PATIENT==feat_path.split("/")[-1].split(".h5")[0]][self.target_label].values[0]]
        #            for feat_path in self.feat_list])
        #return len(targets[targets==1])
        indices = indices if indices is not None else self.indices
        targets = np.array(self.get_targets(indices))
        return np.sum(targets == 1)

    def get_nr_neg(self, indices=None):
        #targets = np.array([self.target_dict[self.df[self.df.PATIENT==feat_path.split("/")[-1].split(".h5")[0]][self.target_label].values[0]]
        #            for feat_path in self.feat_list])
        #return len(targets[targets==0])
        indices = indices if indices is not None else self.indices
        targets = np.array(self.get_targets(indices))
        return np.sum(targets == 0)

    def get_patient_ids(self):
        return [self.df[self.df.PATIENT == "-".join(feat_path.split("/")[-1].split(".h5")[0].split("-")[:3])].PATIENT.values[0]
                    for feat_path in self.feat_list]
                    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        feat_path = self.feat_list[actual_idx]
        pat = "-".join(feat_path.split("/")[-1].split(".h5")[0].split("-")[:3])
        target = self.target_dict[self.df[self.df.PATIENT==pat][self.target_label].values[0]]
        feats = torch.from_numpy(h5py.File(feat_path)["features"][:])
        if feats.shape[0] > self.nr_feats:
            indices = torch.randperm(feats.shape[0])
            feats = feats[indices[:self.nr_feats]]
        else:
            feats = torch.cat((feats, torch.zeros(self.nr_feats - feats.shape[0], feats.shape[1])))
        feats = torch.cat((feats, torch.zeros(self.nr_feats - feats.shape[0], feats.shape[1])))

        return feats, target, pat

def load_lit_model(model_config, checkpoint_path):
    model = LitModel(model_config)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    model.cuda()
    return model

def normalize(feats, state, device="cuda:0"):
    conds_mean = state["conds_mean"][None, None, :].to(device)
    conds_std = state["conds_std"][None,None,:].to(device)
    
    return (feats-conds_mean)/conds_std    

def denormalize(feats, state, device="cuda:0"):
    conds_mean = state["conds_mean"][None, None, :].to(device)
    conds_std = state["conds_std"][None,None,:].to(device)
    
    return (feats*conds_std)+conds_mean

def manipulate(model, feats, state, man_amp=0.6, cls_id=1, device="cuda:0"):
    feats = feats.detach()
    feats.requires_grad = True
    model.zero_grad()
    scores = model(feats)
    scores[:,cls_id].sum().backward()
    normalized_class_direction = F.normalize(feats.grad, dim=2)
    
    normalized_feats = normalize(feats,state,device)
    
    norm_man_amp = man_amp * math.sqrt(512)
    
    norm_man_feats = normalized_feats + norm_man_amp * normalized_class_direction
    
    return denormalize(norm_man_feats,state,device)

def save_image(image_tensor, save_path):
    image = transforms.ToPILImage()(image_tensor.cpu().squeeze(0))
    image.save(save_path)
    
def get_top_tiles(model, feats, k=15, cls_id=1, device='cuda:0'):
    unsq = feats.squeeze(0).unsqueeze(1).to(device)
    scores = F.softmax(model(unsq),dim=1)
    return  scores[:, cls_id].topk(k).indices

def train(model,
          train_set,
          val_set,
          test_set,
          full_dataset,
          out_dir,
          positive_weights,
          conf,
          model_name="PMA_mil.pth",
          ):

    train_loader = DataLoader(train_set, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers)
    val_loader = DataLoader(val_set, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    test_loader = DataLoader(test_set, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    data_loader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=conf.num_workers)

    loader_dict = {"train": train_loader, "val": val_loader, "test": test_loader, "total": data_loader}

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=conf.lr,)

    #positive_weights = torch.tensor(positive_weights[1]/positive_weights[0], dtype=torch.float).cuda()
    positive_weights = torch.tensor(positive_weights, dtype=torch.float).cuda()

    criterion = nn.CrossEntropyLoss(weight=positive_weights)
    #criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weights)

    best_val_loss = float('inf')  # Initialize with a large value
    best_model_state_dict = None
    pbar = tqdm(total=conf.num_epochs, desc='Training Progress', unit='epoch')

    stop_count = 0

    breakpoint = False
    
    for epoch in range(conf.num_epochs):
        model.train()  # Set the model to training mode

        if breakpoint:
            break
        
        total_train_loss = 0.0
        total_train_correct = 0

        total_val_loss = 0.0
        total_val_correct = 0
        
        for mode in ["train","val"]:
        
            for feats, targets, _ in tqdm(loader_dict[mode],leave=False):
                if torch.cuda.is_available():
                    feats = feats.cuda()
                    targets = targets.cuda().to(torch.long)

                if mode == "val":
                    model.eval()
                    with torch.no_grad():
                        logits = model(feats)
                        loss = criterion(logits, targets)
                    total_val_loss += loss.item()
                    _, predicted_labels = torch.max(logits, dim=1)
                    total_val_correct += (predicted_labels == targets).sum().item()
                    
                else:
                    logits = model(feats)
                    loss = criterion(logits, targets)
                    optimizer.zero_grad() 
                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item()

                    _, predicted_labels = torch.max(logits, dim=1)
                    total_train_correct += (predicted_labels == targets).sum().item()
            if mode=="train":
                # Calculate training metrics
                train_loss_avg = total_train_loss / len(train_loader)
                train_accuracy = total_train_correct / len(train_set)
            else:
                # Calculate validation metrics
                val_loss_avg = total_val_loss / len(val_loader)
                val_accuracy = total_val_correct / len(val_set)

                # Checkpointing: Save the model with the lowest validation loss
                if val_loss_avg <= best_val_loss:
                    best_val_loss = val_loss_avg
                    best_model_state_dict = model.state_dict()
                    stop_count = 0
                    tqdm.write('Epoch: {}, loss: {:.4f}, Acc.: {:.2f}%, val_loss: {:.4f}, val_acc: {:.2f}%'.format(
                        epoch + 1, train_loss_avg, train_accuracy * 100, val_loss_avg, val_accuracy * 100))
                else:
                    stop_count += 1
                    if stop_count >= conf.es:
                        print("Early stopping triggered!")
                        breakpoint = True
                        break 
        pbar.set_postfix(
        loss=train_loss_avg,
        acc=train_accuracy,
        val_loss=val_loss_avg,
        val_acc=val_accuracy
        )
        pbar.update(1)

    pbar.close()
    torch.save(best_model_state_dict,f"{out_dir}/{model_name}")
    model.load_state_dict(best_model_state_dict)
    with open(f"{out_dir}/loader.pkl","wb") as f:
        pickle.dump(loader_dict,f)
    
    return model, loader_dict

def test(model, loader_dict, target_label, out_dir, positive_weights):
    
    # Evaluate the model on the test set
    model.eval()
    total_test_correct = 0
    all_predicted_probs = []
    all_targets = []

    pred_dict = {"preds":[], target_label:[]}

    test_loss = 0
    
    #positive_weights = compute_class_weight('balanced', classes=[0,1], y=loader_dict["train"].dataset.get_targets())
    print(f"{positive_weights=}")
    #positive_weights = torch.tensor(positive_weights[1]/positive_weights[0], dtype=torch.float).cuda()
    positive_weights = torch.tensor(positive_weights, dtype=torch.float).cuda()

    criterion = nn.CrossEntropyLoss(weight=positive_weights)
       
    aurocs = []
    
    for feats, targets, _ in tqdm(loader_dict["test"],leave=False):
        if torch.cuda.is_available():
            feats = feats.cuda()
            #targets = nn.functional.one_hot(targets,num_classes=num_classes).cuda().to(torch.int64)
            targets = targets.cuda()

        with torch.no_grad():
            logits = model(feats)
            loss = criterion(logits,targets)
            test_loss += loss.item()
        
            _, predicted_labels = torch.max(logits, dim=1)
            total_test_correct += (predicted_labels == targets).sum().item()

            predicted_probs = nn.functional.softmax(logits, dim=1)
            pred_dict["preds"].extend(predicted_probs[:,1].cpu().numpy().flatten().tolist())
            pred_dict[target_label].extend(targets.cpu().numpy().flatten().tolist())
            all_predicted_probs.append(predicted_probs[:, 1].cpu().numpy())  
            all_targets.append(targets.cpu().numpy())

    # Calculate test accuracy
    test_accuracy = total_test_correct / len(loader_dict["test"].dataset)

    # Flatten the predicted probabilities and targets
    all_predicted_probs = np.concatenate(all_predicted_probs)
    all_targets = np.concatenate(all_targets)

    test_loss_avg = test_loss / len(loader_dict["test"])
    assert len(np.unique([len(pred_dict[k]) for k in pred_dict.keys()]))==1, f"the lengths of the lists are different: {[len(pred_dict[k]) for k in pred_dict.keys()]}"

    test_df = pd.DataFrame(pred_dict)
    test_df.to_csv(f"{out_dir}/PMA_mil_preds_test.csv",index=False)

    # Calculate AUROC
    test_auroc = roc_auc_score(all_targets, all_predicted_probs)
    aurocs.append(test_auroc)

    fpr, tpr, _ = roc_curve(all_targets, all_predicted_probs)

    precision, recall, _ = precision_recall_curve(all_targets, all_predicted_probs)
    
    fig_roc = plt.figure(figsize=(10, 8))
    ax_roc = fig_roc.add_subplot(111)

    colormap=cm.acton_r
    color = colormap(0.5)

    fig_prc = plt.figure(figsize=(10, 8))
    ax_prc = fig_prc.add_subplot(111)
    ax_prc.set_title("Precision-Recall Curve", fontsize=16) 

    ax_roc.plot(fpr, tpr, label=f"AUC={test_auroc:.3f}", color=color)

    ax_prc.plot(recall,precision,label=f"PRC")

    tqdm.write(f"Test loss: {test_loss_avg:.4f}, Test Acc: {test_accuracy*100:.2f}, AUROC: {test_auroc:.4f}")

    ax_roc.legend(loc="lower right", bbox_to_anchor=(0.97, 0.03), prop={'size': 16})
    fig_prc.legend(loc="lower right", bbox_to_anchor=(0.97, 0.03), prop={'size': 16})
    ax_roc.plot([0, 1], [0, 1], "gray", linestyle="--")
    ax_prc.tick_params(axis='both', which='both', labelsize=14)
    ax_roc.tick_params(axis='both', which='both', labelsize=14)

    ax_roc.fill_between(fpr, tpr, alpha=0.05, color=color)


    ax_prc.set_xlabel('Recall',fontsize=14)
    ax_prc.set_ylabel('Precision',fontsize=14)
    ax_roc.set_xlabel('False Positive Rate', fontproperties=my_font, fontsize=14)
    ax_roc.set_ylabel('True Positive Rate', fontproperties=my_font, fontsize=14)
    ax_roc.set_aspect("equal")
    ax_roc.set_title(f'AUC = {np.mean(aurocs):.3f}$\pm${np.std(aurocs):.3f}', fontsize=16)

    fig_roc.savefig(f"{out_dir}/ROC-mil-{target_label}.pdf",dpi=300)
    fig_roc.savefig(f"{out_dir}/ROC-mil-{target_label}.png", dpi=300)
    fig_roc.savefig(f"{out_dir}/ROC-mil-{target_label}.svg")
    fig_prc.savefig(f"{out_dir}/PRC-mil-{target_label}.pdf",dpi=300)
    
    # Save all predictions to see if manipulation will have an effect
    
    pred_dict = {"pat_ids":loader_dict["total"].dataset.get_patient_ids(),"preds":[], target_label:[]}
    all_predicted_labels = []
    # all_targets = []
    
    for feats, targets, _ in tqdm(loader_dict["total"],leave=False):
        if torch.cuda.is_available():
            feats = feats.cuda()
            #targets = nn.functional.one_hot(targets,num_classes=num_classes).cuda().to(torch.int64)
            targets = targets.cuda()

        with torch.no_grad():
            logits = model(feats)
            #loss = criterion(logits,targets)
            #test_loss += loss.item()
        
            _, predicted_labels = torch.max(logits, dim=1)
            #total_test_correct += (predicted_labels == targets).sum().item()

            predicted_probs = nn.functional.softmax(logits, dim=1)
            pred_dict["preds"].extend(predicted_probs[:,1].cpu().numpy().flatten().tolist())
            pred_dict[target_label].extend(targets.cpu().numpy().flatten().tolist())
            all_predicted_labels.append(predicted_labels.cpu().numpy().flatten().tolist())
            # all_predicted_probs.append(predicted_probs[:, 1].cpu().numpy())  
            # all_targets.append(targets.cpu().numpy())
    total_pred_labs = np.concatenate(all_predicted_labels)
    total_df = pd.DataFrame(pred_dict)
    total_df.to_csv(f"{out_dir}/PMA_mil_preds_total.csv",index=False)
    print(f"Total nr predictions: {len(total_df)}; positives: {len(total_pred_labs[total_pred_labs==1])}")