import pandas as pd
import numpy as np
import cv2
import clip
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import umap.umap_ as umap
from mpl_toolkits.mplot3d import Axes3D
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from torch.nn import Softmax
import torch.nn.functional as F
import pickle 

classes_dict = {
0:"screwdriver",
1:"comb",
2:"knife",
3:"pen",
4:"toothbrush"
}

def calculate_median_embeddings(path, model, preprocess, device):

    all_embeddings = {}

    for obj in os.listdir(path):

        ctx_root = os.path.join(path, obj, "context")
        if not os.path.isdir(ctx_root):
            continue

        print("Processing:", obj)

        for ctx in os.listdir(ctx_root):

            ctx_path = os.path.join(ctx_root, ctx)
            embs = []

            for img_name in os.listdir(ctx_path):
                img = Image.open(os.path.join(ctx_path, img_name)).convert("RGB")
                img = preprocess(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model.encode_image(img)
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    embs.append(emb.cpu().numpy())

            all_embeddings[f"{obj}_{ctx}"] = np.vstack(embs)

    median_embeddings = {
        k: np.median(v, axis=0) for k, v in all_embeddings.items()
    }

    median_keys = list(median_embeddings.keys())
    median_matrix = np.stack([median_embeddings[k] for k in median_keys])

    with open("median_matrix.pkl", "wb") as f:
        pickle.dump(median_matrix, f)

    print("Saved median_matrix.pkl")

    return median_embeddings, median_matrix

class detection_module:

    def __init__(self, model):
        self.model = model

    def test(self, img):
        results = self.model(img, conf=0.25)
        result_list = []

        for r in results:
            if r.obb is None:
                continue

            obb = r.obb

            result_dict = {
                "cls": obb.cls.cpu().numpy().astype(int),
                "conf": obb.conf.cpu().numpy(),
                "obb": obb.xyxyxyxy.cpu().numpy()
            }

            result_list.append(result_dict)

        return result_list

#Helper function to get CLIP embeddings from image
def clip_embed_image(img_np, model, preprocess, device):
    """
    img_np: np.ndarray (H,W,3)
    returns: torch.Tensor (D,)
    """
    img_pil = Image.fromarray(img_np).convert("RGB")
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.squeeze(0)


class hierarchical_context:

    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.model.eval()

    def get_best_context(self, img, detections_list):
        """
        returns:
        dict {object_idx: {
            "cls": int,
            "best_context_emb": torch.Tensor (D,),
            "best_context_type": str,
            "score": float
        }}
        """

        H, W = img.shape[:2]
        assert H == 896 and W == 896

        results = {}


        if len(detections_list) == 1:
            det = detections_list[0]

            full_resized = cv2.resize(img, (224, 224))
            full_emb = clip_embed_image(
                full_resized, self.model, self.preprocess, self.device
            )

            results[0] = {
                "cls": det["cls"],
                "best_context_emb": full_emb,
                "best_context_type": "full",
                "score": 1.0
            }
            return results


        for idx, det in enumerate(detections_list):

            obb = det["obb"]
            cls_id = det["cls"]

            xs = [p[0] for p in obb]
            ys = [p[1] for p in obb]

            xmin, xmax = int(min(xs)), int(max(xs))
            ymin, ymax = int(min(ys)), int(max(ys))

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(W, xmax), min(H, ymax)

            obj_crop = img[ymin:ymax, xmin:xmax]
            obj_crop = cv2.resize(obj_crop, (224, 224))

            obj_emb = clip_embed_image(
                obj_crop, self.model, self.preprocess, self.device
            )

          
            contexts = {
                "col": img[:, xmin:xmax],
                "row": img[ymin:ymax, :],
                "full": img
            }

            bw, bh = xmax - xmin, ymax - ymin
            cx, cy = (xmin + xmax)//2, (ymin + ymax)//2

            exmin = max(0, cx - bw)
            exmax = min(W, cx + bw)
            eymin = max(0, cy - bh)
            eymax = min(H, cy + bh)

            contexts["box2x"] = img[eymin:eymax, exmin:exmax]

            best_score = -1
            best_emb = None
            best_name = None

            for name, ctx in contexts.items():
                ctx_resized = cv2.resize(ctx, (224, 224))
                ctx_emb = clip_embed_image(
                    ctx_resized, self.model, self.preprocess, self.device
                )

                score = torch.dot(obj_emb, ctx_emb).item()

                if score > best_score:
                    best_score = score
                    best_emb = ctx_emb
                    best_name = name

            results[idx] = {
                "cls": cls_id,
                "best_context_emb": best_emb, 
                "best_context_type": best_name,
                "score": best_score
            }

        return results

def Hopfield_update(beta=8, memory_matrix=None, query_emb=None, device="cpu"):
    """
    memory_matrix: np.ndarray or torch.Tensor (N, D)
    query_emb: np.ndarray or torch.Tensor (D,)
    """

    assert memory_matrix is not None and query_emb is not None

    
    if isinstance(memory_matrix, np.ndarray):
        X = torch.from_numpy(memory_matrix).float().to(device)
    else:
        X = memory_matrix.clone().detach().float().to(device)

    
    if isinstance(query_emb, np.ndarray):
        q = torch.from_numpy(query_emb).float().to(device)
    else:
        q = query_emb.clone().detach().float().to(device)

    # Ensure shape (D,)
    q = q.view(-1)


    scores = beta * torch.matmul(X, q)     # (N,)
    weights = F.softmax(scores, dim=0)     # (N,)
    xi_new = torch.matmul(weights, X)      # (D,)

    return xi_new

def tokenize_texts(texts, device):
    if isinstance(texts, str):
        texts = [texts]

    text_tokens = clip.tokenize(texts).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)

    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:",device)
    print("Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    
    YOLO_WEIGHTS = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\obb\train4\weights\best.pt"

    MEDIAN_EMBEDDINGS_PATH = r"C:\Users\surya\Desktop\computer vision\Hopfield networks\median_matrix.pkl"
    DATA_PATH =  r"C:\Users\surya\Desktop\computer vision\Hopfield networks\data_uniform"

    print("Loading Context memory...")
    if not os.path.exists(MEDIAN_EMBEDDINGS_PATH):

        calculate_median_embeddings(DATA_PATH,model,preprocess,device)

    with open(MEDIAN_EMBEDDINGS_PATH, 'rb') as f:
        MEMORY_BANK = pickle.load(f)   

    print("Loading YOLO Weights...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    print("Loading image...")
    image = cv2.imread(r"C:\Users\surya\Desktop\computer vision\Hopfield networks\test.jpg")
    image = cv2.resize(image,(896,896))

    detector = detection_module(yolo_model)
    results = detector.test(image)

    print("Getting best context embeddings for each object...")
    hier_context_module = hierarchical_context(model,preprocess,device)
    best_contexts = hier_context_module.get_best_context(image,results)

    # Prepare Image for Plotting
    output_image = image.copy()

    for context_key,context_value in best_contexts.items():

        cls_id = int(context_value["cls"][0])
        context_emb = context_value["best_context_emb"]

        pred_label = classes_dict.get(int(cls_id),"unknown")

        pred_label_text_embedding = tokenize_texts(pred_label,device) 

        retrived_context_emb = Hopfield_update(
        memory_matrix=MEMORY_BANK,
        query_emb=context_emb,
        device=device)

        retrived_pred_emb = Hopfield_update(
            memory_matrix=MEMORY_BANK,
            query_emb=pred_label_text_embedding.squeeze(0),
            device=device)

        sim = cosine_similarity(retrived_context_emb.cpu().numpy().reshape(1,-1),retrived_pred_emb.cpu().numpy().reshape(1,-1))
        
        # --- Visualization Logic ---
        
        # Retrieve the original detection points from 'results' list using the key
        raw_obb = results[context_key]["obb"]
        pts = raw_obb.astype(np.int32).reshape((-1, 1, 2))

        score_val = sim[0,0]
        
        if score_val > 0.9:
            status_text = f"{pred_label}: CORRECT ({score_val:.2f})"
            box_color = (0, 255, 0) # Green in BGR
            text_color = (0, 255, 0)
        else:
            status_text = f"{pred_label}: BAD CONTEXT ({score_val:.2f})"
            box_color = (0, 0, 255) # Red in BGR
            text_color = (0, 0, 255)

        # Draw the Oriented Bounding Box
        cv2.polylines(output_image, [pts], isClosed=True, color=box_color, thickness=3)

        # Draw Text Label (placed slightly above the box)
        # Find the top-most point to place text
        top_point = np.min(pts, axis=0)[0]
        text_pos = (max(0, top_point[0]), max(20, top_point[1] - 10))
        
        cv2.putText(output_image, status_text, text_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        print(f"Processed {pred_label} | Score: {score_val:.3f}")

    # Show final image using Matplotlib
    plt.figure(figsize=(10, 10))
    # Convert BGR to RGB for matplotlib
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Context Aware Object Verification")
    plt.show()