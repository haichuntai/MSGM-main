import os.path as osp
import warnings

from mmcv.runner import Hook
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from mmdet.utils import get_dist_info
from mmdet.utils import all_gather_tensor, synchronize
import mmcv
import numpy as np
import sys


class ExtractFeatureHook(Hook):
    """Evaluation hook.

    Notes:
        If new arguments are added for EvalHook, tools/test.py may be
    effected.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        start (int, optional): Evaluation starting epoch. It enables evaluation
            before the training starts if ``start`` <= the resuming epoch.
            If None, whether to evaluate is merely decided by ``interval``.
            Default: None.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, dataloader, start=None, interval=1, logger=None, pretrained_feature_file=None, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.dataloader = dataloader
        self.logger = logger
        self.pretrained_feature_file = pretrained_feature_file

    def before_run(self, runner):
    # def before_train_epoch(self, runner):
        self.logger.info('start feature extraction for hybgrid memory initialization')
        
        # return

        with torch.no_grad():
            print('feature extract from: ', self.pretrained_feature_file)
            features, features_crop, img_ids, gt_labels = self.extract_features(
                runner.model, self.dataloader, self.dataloader.dataset, with_path=False, prefix="Extract: ", \
                pretrained_feature_file=self.pretrained_feature_file)
            assert features.size(0) == self.dataloader.dataset.id_num
            assert img_ids.size(0) == self.dataloader.dataset.id_num

            # mmcv.dump(img_ids, 'cuhk_img_ids.pkl')
            # mmcv.dump(gt_labels, 'cuhk_gt_labels.pkl')
            # sys.exit()
            
            # f_epoch = 1
            # f_name = 'prw_dicl_0530_4'
            # features = mmcv.load('work_dirs/{}/memory_featrues/roi/epoch_{}.pkl'.format(f_name, f_epoch)) # torch.Size([18048, 256])
            # features_crop = mmcv.load('work_dirs/{}/memory_featrues/crop/epoch_{}.pkl'.format(f_name, f_epoch))
            # img_ids = mmcv.load('img_ids.pkl')
            # gt_labels = mmcv.load('gt_labels.pkl') # 一共184个标签

            # if features_crop is None:
            #     features_crop = features
            runner.model.module.roi_head.bbox_head.loss_reid._update_feature(features)
            runner.model.module.roi_head.bbox_head.loss_reid._init_ids(img_ids) # torch.Size([18048])
            runner.model.module.roi_head.bbox_head.loss_reid._init_gt_labels(gt_labels)
            runner.model.module.roi_head.bbox_head.loss_reid_crop._update_feature(features_crop)
            runner.model.module.roi_head.bbox_head.loss_reid_crop._init_ids(img_ids)
            runner.model.module.roi_head.bbox_head.loss_reid_crop._init_gt_labels(gt_labels)

            # part memoty 初始化
            runner.model.module.roi_head.bbox_head.loss_reid_crop_p1._update_feature(features_crop.clone().detach())
            runner.model.module.roi_head.bbox_head.loss_reid_crop_p1._init_ids(img_ids)
            runner.model.module.roi_head.bbox_head.loss_reid_crop_p1._init_gt_labels(gt_labels)
            runner.model.module.roi_head.bbox_head.loss_reid_crop_p2._update_feature(features_crop.clone().detach())
            runner.model.module.roi_head.bbox_head.loss_reid_crop_p2._init_ids(img_ids)
            runner.model.module.roi_head.bbox_head.loss_reid_crop_p2._init_gt_labels(gt_labels)


    @torch.no_grad()
    def extract_features(self,
        model,  # model used for extracting
        data_loader,  # loading data
        dataset,  # dataset with file paths, etc
        cuda=True,  # extract on GPU
        normalize=True,  # normalize feature
        with_path=False,  # return a dict {path:feat} if True, otherwise, return only feat (Tensor)  # noqa
        print_freq=10,  # log print frequence
        save_memory=False,  # gather features from different GPUs all together or in sequence, only for distributed  # noqa
        for_testing=True,
        prefix="Extract: ",
        pretrained_feature_file=None,
    ):


        rank, world_size, is_dist = get_dist_info()
        features = []

        model.eval()
        try:
            if isinstance(model.module.roi_head.bbox_head, nn.ModuleList):
                print('------')
                for i in range(len(model.module.roi_head.bbox_head)):
                    model.module.roi_head.bbox_head[i].proposal_score_max=True
            else:
                print('****')
                model.module.roi_head.bbox_head.proposal_score_max=True
                ori_iou_threshold = model.module.roi_head.test_cfg.nms.iou_threshold
                model.module.roi_head.test_cfg.nms.iou_threshold=2
                ori_max_per_img = model.module.roi_head.test_cfg.max_per_img
                model.module.roi_head.test_cfg.max_per_img=1000
        except:
            assert False, "setting fault"
            pass
        # data_loader.dataset.load_fake_proposals()
        data_iter = iter(data_loader)

        #prog_bar = mmcv.ProgressBar(len(data_loader))

        # features = torch.zeros(dataset.id_num, 256)
        features = None
        features_crop = None
        img_ids = torch.zeros(dataset.id_num).long()
        gt_labels = torch.zeros(dataset.id_num).long()

        if pretrained_feature_file is not None:
            pretrain_features = mmcv.load(pretrained_feature_file)

        prog_bar = mmcv.ProgressBar(len(data_loader))
        for i in range(len(data_loader)):
            data = next(data_iter)
            # print(type(data))
            # print(data.keys())
            #print(data['gt_labels'])
            #print(data['gt_bboxes'])
            #print(data['img_metas'])
            # print(data['img'])
            gt_bboxes=data['gt_bboxes'][0]._data[0][0]
            gt_ids = data['gt_labels'][0]._data[0][0][:, 1] 
            gt_img_ids = data['gt_labels'][0]._data[0][0][:, 2]
            gt_label = data['gt_labels'][0]._data[0][0][:, 3]

            if pretrained_feature_file is not None: # 0
                print('pretrained_feature_file')
                if features is None:
                    features = torch.zeros(dataset.id_num, pretrain_features[i][0].shape[1]-5)
                pretrained_gt_bboxes=pretrain_features[i][0][:, :4]
                pretrained_gt_bboxes=torch.from_numpy(pretrained_gt_bboxes)
                # gt_bboxes[:, 2:]=gt_bboxes[:, 2:] + gt_bboxes[:, :2]
                # print(data['img_metas'][0][0])
                scale_factor = data['img_metas'][0].data[0][0]['scale_factor']
                scale_factor=torch.from_numpy(scale_factor).unsqueeze(dim=0)
                pretrained_gt_bboxes = pretrained_gt_bboxes*scale_factor
                diff=(pretrained_gt_bboxes - gt_bboxes).abs().sum()
                if diff>1:
                    print("pretrained boxes don't match")
                    print(diff)
                    exit()
                # print(gt_ids)
                features[gt_ids] = torch.from_numpy(pretrain_features[i][0][:, 5:])
                img_ids[gt_ids] = gt_img_ids
                prog_bar.update()
                continue
            new_data = {'proposals':data['gt_bboxes'], 'img': data['img'], 'img_metas': data['img_metas']}
            result = model(return_loss=False, rescale=False, use_gtx=True, **new_data) # (2, 261) [0][0]

            c = 256
            reid_features = torch.from_numpy(result[0][0][:, 5:5+c]) # torch.Size([2, 256])
            reid_features_crop = torch.from_numpy(result[0][0][:, 5+c:]) # torch.Size([2, 256])
            #result = np.asarray(result)
            #reid_features = torch.from_numpy(result[:, 0, 0, 5:])
            if normalize:
                reid_features = F.normalize(reid_features, p=2, dim=-1)
                reid_features_crop = F.normalize(reid_features_crop, p=2, dim=-1)
            
            if features is None:
                features = torch.zeros(dataset.id_num, reid_features.shape[1])
            if features_crop is None:
                features_crop = torch.zeros(dataset.id_num, reid_features_crop.shape[1])
            
            #align gt box and predicted box
            result_boxes = torch.from_numpy(result[0][0][:, :4])
            if result_boxes.shape != gt_bboxes.shape: # 0
                # print(model.module.roi_head.test_config.nms.iou_threshold)
                print(result_boxes.shape, gt_bboxes.shape)
                print(result_boxes)
                print(gt_bboxes)
                print(result[0][0][:, :5])
                exit()
            
            result_boxes = result_boxes.unsqueeze(dim=1) # torch.Size([2, 1, 4])
            gt_bboxes = gt_bboxes.unsqueeze(dim=0) # torch.Size([1, 2, 4])
            #print(result_boxes, gt_bboxes)
            diff = (result_boxes - gt_bboxes).abs().sum(dim=-1) # torch.Size([2, 2])
            #print(diff)
            minis = diff.argmin(dim=-1) # 挑选跟真实标注框比较相像的检测框
            #print('minis', minis)
            # print(diff.min(dim=-1))
            gt_ids = gt_ids[minis]
            
            features[gt_ids] = reid_features
            features_crop[gt_ids] = reid_features_crop
            img_ids[gt_ids] = gt_img_ids
            gt_labels[gt_ids] = gt_label
            # print('=---------')
            # if i>10:
                # exit()
            # continue
            
            # for i in range(result_boxes.shape[0]):
            #     result_box = result_boxes[i:i+1, :]
            #     mini = (gt_bboxes - result_box).abs().sum(dim=1).argmin()
            #     gt_id = gt_ids[mini]
            #     # print(gt_id)
            #     features[gt_id] = reid_features[i]
            prog_bar.update()
            
            # break
        
        #restore model status
        model.train()
        try:
            if isinstance(model.module.roi_head.bbox_head, nn.ModuleList):
                for i in range(len(model.module.roi_head.bbox_head)):
                    model.module.roi_head.bbox_head[i].proposal_score_max=False
            else:
                model.module.roi_head.bbox_head.proposal_score_max=False
                model.module.roi_head.test_cfg.nms.iou_threshold=ori_iou_threshold
                model.module.roi_head.test_cfg.max_per_img=ori_max_per_img
        except:
            assert False, "restore setting fault"
            pass

        synchronize()

        if is_dist and cuda: # 0
            # distributed: gather features from all GPUs
            all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
            all_features = all_features.cpu()[: len(dataset)]
            all_img_ids = all_gather_tensor(img_ids.cuda(), save_memory=save_memory)
            all_img_ids = all_img_ids.cpu()[: len(dataset)]
        else:
            all_features_crop = features_crop
            all_features = features
            all_img_ids = img_ids
        print('all_features', all_features.shape)
        return all_features, all_features_crop, img_ids, gt_labels # torch.Size([18048, 256]), torch.Size([18048])
