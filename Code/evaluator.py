import torch
import speech_recognition as sr
import pyaudio
from torch import nn
import numpy as np
import PIL
import subprocess
import os                                                     

from anchors import (create_anchors, reg_params_to_bbox,
                     IoU_values, x1y1x2y2_to_y1x1y2x2)
from typing import Dict
from functools import partial
from pydub import AudioSegment
from pydub.playback import play
# from utils import reduce_dict


def reshape(box, new_size):
    """
    box: (N, 4) in y1x1y2x2 format
    new_size: (N, 2) stack of (h, w)
    """
    box[:, :2] = new_size * box[:, :2]
    box[:, 2:] = new_size * box[:, 2:]
    return box


class Evaluator(nn.Module):
    """
    To get the accuracy. Operates at training time.
    """

    def __init__(self, ratios, scales, cfg):
        super().__init__()
        self.cfg = cfg

        self.ratios = ratios
        self.scales = scales

        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']
        self.use_focal = cfg['use_focal']
        self.use_softmax = cfg['use_softmax']
        self.use_multi = cfg['use_multi']

        self.lamb_reg = cfg['lamb_reg']

        self.met_keys = ['Acc', 'MaxPos']
        self.anchs = None
        self.get_anchors = partial(
            create_anchors, ratios=self.ratios,
            scales=self.scales, flatten=True)

        self.acc_iou_threshold = self.cfg['acc_iou_threshold']

    def forward(self, out: Dict[str, torch.tensor],
                inp: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:

        annot = inp['annot']
        att_box = out['att_out']

        reg_box = out['bbx_out']

        feat_sizes = out['feat_sizes']

        num_f_out = out['num_f_out']
 
        device = att_box.device

        if len(num_f_out) > 1:
            num_f_out = int(num_f_out[0].item())
        else:
            num_f_out = int(num_f_out.item())

        feat_sizes = feat_sizes[:num_f_out, :]

        if self.anchs is None:
            feat_sizes = feat_sizes[:num_f_out, :]
            anchs = self.get_anchors(feat_sizes)
            anchs = anchs.to(device)
            self.anchs = anchs
        else:
            anchs = self.anchs

        att_box_sigmoid = torch.sigmoid(att_box).squeeze(-1)
        att_box_best, att_box_best_ids = att_box_sigmoid.max(1)
        topk_box = torch.topk(att_box_sigmoid,k=100) #added by rishabh
        att_box_best, att_box_best_ids = topk_box.values, topk_box.indices  #added by rishabh

        ious1 = IoU_values(annot, anchs)
        #print("-->iou:", ious1.shape)
        gt_mask, expected_best_ids = ious1.max(1)

        actual_bbox = reg_params_to_bbox(
            anchs, reg_box)

        #print("\n ->actual_bbox", actual_bbox[0][att_box_best_ids[0].item()])
        best_possible_result, _ = self.get_eval_result(
            actual_bbox, annot, expected_best_ids)

        #print("\n--> att_box_best_ids:", att_box_best_ids)

        msk = None

        #'''
        pred_box = self.get_eval_result(
                actual_bbox, annot, att_box_best_ids[:,0], msk)[1]

        top_boxes = x1y1x2y2_to_y1x1y2x2(reshape(
            (pred_box + 1)/2, (inp['img_size'])))

        top_scores = att_box_best[:,0]
        for i in range(1,100,1):
            #Rishabh--Break if att_box_best[:,i] < 0.5
            if att_box_best[:,i] < 0.45:
                break

            actual_result, pred_boxes = self.get_eval_result(
                actual_bbox, annot, att_box_best_ids[:,i], msk)
            #'''

           #actual_result, pred_boxes = self.get_eval_result(
           #     actual_bbox, annot, att_box_best_ids, msk)

            #print("\n-> pred box", att_box_best_ids, pred_boxes)
            '''
            print("R-------In Eevaluator.py",
                "\n annot", annot,
                "\n att_box", att_box, att_box.shape,
                "\n reg_box", reg_box, reg_box.shape,
                "\n anchs", anchs, anchs.shape,
                "\n actual_result", actual_result, 
                "\n pred_boxes", pred_boxes, 
                "\n best_possible_result", best_possible_result,
                "\n actual_bbox", actual_bbox, actual_bbox.shape,
                "\n att_box_best", att_box_best)
            '''
            #ris
            out_dict = {}
            out_dict['Acc'] = actual_result
            out_dict['MaxPos'] = best_possible_result
            out_dict['idxs'] = inp['idxs']

            iou_sc = IoU_values(pred_boxes, anchs)
            reshaped_boxes = x1y1x2y2_to_y1x1y2x2(reshape(
                (pred_boxes + 1)/2, (inp['img_size'])))

            out_dict['pred_boxes'] = reshaped_boxes
            out_dict['pred_scores'] = att_box_best

            #print("\n\n --> {}-th reshaped_box \n score:{} and \n box:{} iou:{}".format(i, att_box_best[:,i],reshaped_boxes, iou_sc.max(1)))
            #print(IoU_values(top_boxes, reshaped_boxes) < 0.5)
            if not False in (IoU_values(top_boxes, reshaped_boxes) < 0.5):
                top_boxes=torch.cat([top_boxes,reshaped_boxes], axis=0)
                top_scores = torch.cat([top_scores,att_box_best[:,i]], axis=0)
            #print("\n\n --> {}-th reshaped_box \n score:{} and \n box:{} iou:{}".format(i, att_box_best,reshaped_boxes, iou_sc.max(1)))
            # orig_annot = inp['orig_annot']
            # Sanity check
            # iou1 = (torch.diag(IoU_values(reshaped_boxes, orig_annot))
            #         >= self.acc_iou_threshold).float().mean()
            # assert actual_result.item() == iou1.item()


        print("Best bounding boxes--------->\n\n", top_boxes, top_scores)
        import cv2
        import pandas as pd

        #Written by Kritika
        test_dat = pd.read_csv("data/referit/csv_dir/test.csv")
        img_count = 0
        for ind in range(len(test_dat)):
            bb_data = top_boxes.cpu().numpy()#genfromtxt('bb.csv', delimiter=',')
            for box_num in range(0, bb_data.shape[0]):
                filename = test_dat.iloc[ind]['img_id']
                #print(filename)
                #Kritika
                img_path=r'input/{}'.format('img1.jpeg')
                #img_path=r'data/referit/saiapr_tc12_images/{}'.format('imageSend1.jpeg')
                #img_path=r'data/referit/saiapr_tc12_images/{}'.format(filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img,0)
                #img = cv2.rectangle(img,(173, 182), (448, 360),(0,255,0),2)
                #pred_box = test_dat.iloc[ind]['pred_box']
                x1 = round(bb_data[box_num, 0])
                y1 = round(bb_data[box_num, 1])
                x2 = round(bb_data[box_num, 2])
                y2 = round(bb_data[box_num, 3])
                img = cv2.rectangle(img,(x1, y1), (x2, y2),(0,0,255),20)
                img_count = img_count + 1
                nameToRename = "output/imgR" + str(img_count) + '.jpeg'
                cv2.imwrite(nameToRename, img)
                p = subprocess.Popen(["display", nameToRename])
                if  bb_data.shape[0] > 1:
                    found_obj = input("Do you mean this?: ")
                    '''
                    r = sr.Recognizer()
                    with sr.Microphone() as source:
                        print("Say something!")
                        audio = r.listen(source)

                    # Speech recognition using Google Speech Recognition
                    try:
                        # for testing purposes, we're just using the default API key
                        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
                        # instead of `r.recognize_google(audio)`
                        #print("You said: " + r.recognize_google(audio))
                        found_obj = r.recognize_google(audio)
                        print(found_obj)
                    except sr.UnknownValueError:
                        print("Google Speech Recognition could not understand audio")
                    except sr.RequestError as e:
                        print("Could not request results from Google Speech Recognition service; {0}".format(e))
                        '''
                    p.kill()
                    if "y" == found_obj[0].lower():
                        break
                    else:
                        continue
                else:
                    print("Thank you for your confirmation")
        #print(top_boxes.cpu().numpy())
        #np.savetxt("bb.csv", top_boxes.cpu().numpy(), delimiter=",")

        return out_dict
        # return reduce_dict(out_dict)

    def get_eval_result(self, actual_bbox, annot, ids_to_use, msk=None):
        #print("\n --> ids_to_use:", ids_to_use.view(-1, 1, 1).expand(-1, 1, 4))
        best_boxes = torch.gather(
            actual_bbox, 1, ids_to_use.view(-1, 1, 1).expand(-1, 1, 4))
        best_boxes = best_boxes.view(best_boxes.size(0), -1)
        if msk is not None:
            best_boxes[msk] = 0
        # self.best_boxes = best_boxes
        ious = torch.diag(IoU_values(best_boxes, annot))
        #print("best_bbox",best_boxes)
        #print("annot",annot)
        #print("--->Ious in eval--------------------", ious)
        # self.fin_results = ious
        return (ious >= self.acc_iou_threshold).float().mean(), best_boxes


def get_default_eval(ratios, scales, cfg):
    return Evaluator(ratios, scales, cfg)
