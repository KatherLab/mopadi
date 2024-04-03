# %%
import os
from pathlib import Path
from mil.utils import *
from configs.templates_cls import lung_mil
from dotenv import load_dotenv

load_dotenv()
ws_path = os.getenv("WORKSPACE_PATH")


out_dir = f"{ws_path}/mopadi/checkpoints/lung-tim"
man_img_dir = f"{out_dir}/manipulated_imgs"
state_dict_path = f"{out_dir}/PMA_mil.pth"
latent_infer_path = f"{ws_path}/mopadi/checkpoints/pancancer/latent.pkl"
mopadi_path = f"{ws_path}/mopadi/checkpoints/pancancer/last.ckpt"

conf = lung_mil()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if not os.path.exists(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

if not os.path.exists(man_img_dir):
    Path(man_img_dir).mkdir(parents=True, exist_ok=True)

man_amp = 0.03
T_step=100
T_inv=200

model = Classifier(dim=conf.dim, num_heads=conf.num_heads, num_seeds=conf.num_seeds, num_classes=conf.num_classes)
weights = torch.load(state_dict_path)
model.load_state_dict(weights)

if torch.cuda.is_available():
    model = model.cuda()

latent_state = torch.load(latent_infer_path)
#conds = latent_state['conds']

# %% 
with open(f"{out_dir}/loader.pkl","rb") as f:
    loader_dict = pickle.load(f)

model.eval()

#mopadi = load_lit_model(conf, mopadi_path)

target_label = "Type"

# %%
pred_dict = {"pat_ids":loader_dict["total"].dataset.get_patient_ids(),"preds":[], target_label:[]}
all_predicted_labels = []

for feats, targets, pats in tqdm(loader_dict["total"]):
    if torch.cuda.is_available():
        feats = feats.cuda()
        targets = targets.cuda().to(torch.long)
    manipulated_feats = manipulate(model,feats,latent_state,man_amp)
    #TODO: get actual batch of images for this stuff
    #stoch_feats = (feats + F.normalize(torch.randn_like(feats)))
    #stoch_feats = F.normalize(torch.randn(3,224,224)).cuda()
    with torch.no_grad():
        logits = model(manipulated_feats)
        #topk_idxs = get_top_tiles(model,manipulated_feats)
        #top_feats = manipulated_feats.index_select(index=topk_idxs,dim=1).squeeze(0)
        #top_st_feats = stoch_feats.index_select(index=topk_idxs,dim=1).squeeze(0)
        _, predicted_labels = torch.max(logits, dim=1)
        
        predicted_probs = F.softmax(logits, dim=1)        
        pred_dict["preds"].extend(predicted_probs[:,1].cpu().numpy().flatten().tolist())
        pred_dict[target_label].extend(targets.cpu().numpy().flatten().tolist())
        all_predicted_labels.append(predicted_labels.cpu().numpy().flatten().tolist())
        
        #print(f"{top_feats.shape=}")
        
        # for i in tqdm(range(len(top_feats)),leave=False):

        #     manipulated_img = mopadi.render(stoch_feats[None,:], top_feats[i][None,:], T=T_step)[0]
        #     non_s_manipulated_img = mopadi.render(torch.zeros_like(stoch_feats)[None,:], top_feats[i][None,:], T=T_step)[0]
        #     #print(f"{manipulated_img.shape=}")
        #     rgb_man_img = convert2rgb(manipulated_img, adjust_scale=False)
        #     rgb_non_s_img = convert2rgb(non_s_manipulated_img, adjust_scale=False)
        #     save_image(rgb_man_img, f"{man_img_dir}/{pats[0]}-stoch-{i}.png")
        #     save_image(rgb_non_s_img, f"{man_img_dir}/{pats[0]}-{i}.png")
        
total_pred_labs = np.concatenate(all_predicted_labels)
total_df = pd.DataFrame(pred_dict)
total_df.to_csv(f"{out_dir}/PMA_mil_manipulated_preds_total.csv",index=False)
print(f"Total nr predictions: {len(total_df)}; positives: {len(total_pred_labs[total_pred_labs==1])}")
# %%
