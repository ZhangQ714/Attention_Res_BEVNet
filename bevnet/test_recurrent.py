import argparse
import os
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from inference import BEVNetRecurrent
from bev_utils import Evaluator
from train_fixture_utils import make_label_vis, get_colormap
from test_sequence_factory import make
from roc import roc_generate
from graph import paint_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--model_file', type=str, required=True, help='Path to the model.')
parser.add_argument('--test_env', type=str, required=True,
                    help='Test environment. See test_sequence_factory.py')
opts = parser.parse_args()

# VISUALIZATION = opts.visualize
VISUALIZATION = False

MODEL_FILE = opts.model_file
TEST_ENV = opts.test_env


model = BEVNetRecurrent(MODEL_FILE)
test_data = make(TEST_ENV)



if model.g.include_unknown:
    ignore_idx = model.g.num_class - 1
else:
    ignore_idx = 255
e = Evaluator(num_classes=model.g.num_class, ignore_label=ignore_idx)

all_label_th = []
all_pred = []
# all_logits_t = []
# all_pred_t = []

for i in tqdm.trange(len(test_data['scan_files'])):
    scan_fn = test_data['scan_files'][i]
    name = os.path.basename(os.path.splitext(scan_fn)[0])
    label_fn = os.path.join(test_data['label_dir'], name + '.png')

    scan = np.fromfile(scan_fn, dtype=np.float32).reshape(-1, 4)
    label = np.array(Image.open(label_fn), dtype=np.uint8)
    label_th = torch.as_tensor(label, device='cuda').long()

    if model.g.include_unknown:
        # Note that `num_class` already includes the unknown label
        label_th[label_th == 255] = model.g.num_class - 1

    logits = model.predict(scan, test_data['costmap_poses'][i])[0]
    pred = torch.argmax(logits, dim=0)

    all_label_th.append(label_th.cpu().numpy())      # 3931 x (512, 512)
    all_pred.append(pred.cpu().numpy())

    # if i > 500:
    #     break
    # all_logits_t.append(logits.T.cpu().numpy())          # 3931 x (5, 512, 512)
    # all_pred_t.append(pred.cpu().numpy())                # 3931 x (512, 512)

    e.append(pred[None], label_th[None])

    if VISUALIZATION:
        cmap = get_colormap(model.g.dataset_type)
        label_vis = make_label_vis(label, cmap)
        pred_vis = make_label_vis(pred.cpu().numpy(), cmap)
        vis = np.concatenate([pred_vis, label_vis], axis=1)
        # cv2.imshow('', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)ety

        resultpath = os.path.split(MODEL_FILE)[0] + '/result-image/'
        imgpath = os.path.join(resultpath, name + '.png')

        if not os.path.exists(resultpath):
            os.makedirs(resultpath)

        cv2.imwrite(imgpath, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

# 混淆矩阵
paint_confusion_matrix(all_label_th, all_pred, opts, paint_unknown=False)

# all_pred_t = torch.stack(all_pred_t)
# n, w, h = all_pred_t.shape
# all_pred_t = all_pred_t.view(n * w * h)
#
# all_logits_t = torch.stack(all_logits_t)
# n, w, h, c = all_logits_t.shape
# all_logits_t = all_logits_t.view(n * w * h, c)
#
# print("a")
#
# roc_generate(np.array(all_pred_t), np.array(all_logits_t), MODEL_FILE)

# 保存label和pred数据
# txtpath = os.path.split(MODEL_FILE)[0] + '/label_pred_txt/'
# if not os.path.exists(txtpath):
#     os.makedirs(txtpath)
#
# with open(txtpath + 'all_label.txt', 'w') as f:
#     np.savetxt(txtpath + 'all_label.txt', np.array(all_label_th[:1000]).flatten(), fmt='%.0f')
#     np.savetxt(txtpath + 'all_label.txt', np.array(all_label_th[1000:2000]).flatten(), fmt='%.0f')
#     np.savetxt(txtpath + 'all_label.txt', np.array(all_label_th[2000:]).flatten(), fmt='%.0f')
# f.close()
# with open(txtpath + 'all_pred.txt', 'w') as f:
#     np.savetxt(txtpath + 'all_pred.txt', np.array(all_pred[:1000]).flatten(), fmt='%.0f')
#     np.savetxt(txtpath + 'all_pred.txt', np.array(all_pred[1000:2000]).flatten(), fmt='%.0f')
#     np.savetxt(txtpath + 'all_pred.txt', np.array(all_pred[2000:]).flatten(), fmt='%.0f')
# f.close()

ious = e.classwiseIoU()
if model.g.include_unknown:
    ious = ious[:-1]
print('ious:', ious)
print('miou:', np.mean(ious))
