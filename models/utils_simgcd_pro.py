from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm

'''
get_kmeans_centroid_for_new_head(): use model_pre to obtain new class head init

compute_prior_old_new_ratio(): use model_cur to predict old and new ratio

both use online session training data of current stage
'''


def get_kmeans_centroid_for_new_head(model, online_session_train_loader, args, device):

    model.to(device)
    model.eval()

    all_feats = []

    args.logger.info('Perform KMeans for new classification head initialization!')
    args.logger.info('Collating features...')
    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            # Pass features through base model and then additional learnable transform (linear layer)
            feats = model[0](images)   # backbone
            feats = torch.nn.functional.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes+args.num_cur_novel_classes, random_state=0).fit(all_feats)
    #preds = kmeans.labels_
    centroids_np = kmeans.cluster_centers_   # (60, 768)
    print('Done!')

    centroids = torch.from_numpy(centroids_np).to(device)
    centroids = torch.nn.functional.normalize(centroids, dim=-1)   # torch.Size([60, 768])
    #centroids = centroids.float()
    with torch.no_grad():
        _, logits = model[1](centroids)   # torch.Size([60, 50])
        max_logits, _ = torch.max(logits, dim=-1)   # torch.Size([60])
        _, proto_idx = torch.topk(max_logits, k=args.num_novel_class_per_session, largest=False)   # torch.Size([10])
        new_head = centroids[proto_idx]   # torch.Size([10, 768])

    return new_head


def compute_prior_old_new_ratio(model, online_session_train_loader, args, device):
    model.to(device)
    model.eval()

    all_preds_list = [] = []
    args.logger.info('Using model_cur (initialized new heads) to predicting labels...')
    # First extract all features
    with torch.no_grad():
        for batch_idx, (images, label, _, _) in enumerate(tqdm(online_session_train_loader)):
            images = images.cuda(non_blocking=True)
            _, logits = model(images)
            all_preds_list.append(logits.argmax(1))

    all_preds = torch.cat(all_preds_list, dim=0)   # NOTE!!!
    args.logger.info('Computing prior old and new ratio...')
    pred_prior_old_ratio = len(all_preds[all_preds<args.num_seen_classes]) / len(all_preds)
    pred_prior_new_ratio = len(all_preds[all_preds>=args.num_seen_classes]) / len(all_preds)
    args.logger.info(f'Pred prior old ratio: {pred_prior_old_ratio:.4f} | Pred prior new ratio: {pred_prior_new_ratio:.4f}')


    pred_prior_ratio_dict = {
        'pred_prior_old_ratio': pred_prior_old_ratio,
        'pred_prior_new_ratio': pred_prior_new_ratio,
    }

    return pred_prior_ratio_dict
