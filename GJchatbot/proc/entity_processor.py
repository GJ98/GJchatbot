import os
from typing import Dict, List
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from GJchatbot.proc.base_processor import BaseProcessor
from GJchatbot.utils.metrics import scoring


class EntityProcessor(BaseProcessor):

    def __init__(self, model: nn.Module):
        """개채명 분류 모델 Training, Inference 함수를 가진 클래스

        Args:
            model (nn.Module): 개채명 분류 모델
        """

        super().__init__(model)
        
        self.model_lr = 1e-4
        self.weight_decay = 1e-4

        self.lr_scheduler_factor = 0.75
        self.lr_scheduler_patience = 10
        self.lr_scheduler_min_lr = 1e-12

        self.epoch = 20

    def fit(self, 
            tr_dataloader: DataLoader, 
            eval_dataloader: DataLoader, 
            label_dict: Dict[int, str]):
        """개채명 분류 모델 Training 함수
    
        Args:
            tr_dataloader (DataLoader): 학습 데이터
            eval_dataloader (DataLoader): 검증 데이터
            label_dict (Dict[int, str]): 라벨 딕셔너리
        """
        
        if label_dict != self.model.label_dict:
            raise Exception("(data label dict) is not (model label dict)."
                            "data label dictionary: {}"
                            "model label dictionary: {}"
                            .format(label_dict, self.model.label_dict))

        device = torch.device('cuda') if torch.cuda.is_available() else torch.deivce('cpu')

        loss_fn = nn.CrossEntropyLoss()

        optimizer = Adam(params=self.model.parameters(),
                         lr=self.model_lr,
                         weight_decay=self.weight_decay)

        lr_scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                         factor=self.lr_scheduler_factor,
                                         min_lr=self.lr_scheduler_min_lr,
                                         patience=self.lr_scheduler_patience)

        for epoch in tqdm(range(self.epoch), desc='epoch', total=self.epoch):
            tqdm.write("\nepoch: {}, lr: {}".format(epoch, 
                                                    optimizer.param_groups[0]['lr']))
            tr_loss, tr_acc, total_num = 0, 0, 0
            self.model.to(device)
            self.model.train()
            
            for data in tqdm(tr_dataloader, desc='step', total=len(tr_dataloader)):
                optimizer.zero_grad()

                question, label = map(lambda elm: elm.to(device), data)
                pred_label = self.model(question)

                # loss 계산을 위해 shape 변경
                pred_label = pred_label.reshape(-1, pred_label.size(-1))
                label = label.view(-1).long()

                loss = loss_fn(pred_label, label)
                loss.backward()
                optimizer.step()
                #lr_scheduler.step(loss)

                with torch.no_grad():
                    _correct_num, _total_num = scoring(pred_label, label)
                    tr_loss += loss.item()
                    tr_acc += _correct_num
                    total_num += _total_num

            tr_loss_avg = tr_loss / len(tr_dataloader)
            tr_acc_avg = tr_acc / total_num

            tqdm.write("\nepoch: {}, tr_loss: {}, tr_acc: {}".format(epoch,
                                                                     tr_loss_avg,
                                                                     tr_acc_avg))

        print("eval")
        eval_loss, eval_acc, total_num = 0, 0, 0
        for data in tqdm(eval_dataloader, desc='step', total=len(eval_dataloader)):
            self.model.eval()
            question, label = map(lambda elm: elm.to(device), data)

            with torch.no_grad():
                pred_label = self.model(question)

                pred_label = pred_label.view(-1, pred_label.size(-1))
                label = label.view(-1).long()

                # acc 
                _correct_num, _total_num = scoring(pred_label, label)
                eval_acc += _correct_num
                total_num += _total_num

                # loss
                eval_loss += loss_fn(pred_label, label).item() 

        eval_loss_avg = eval_loss / len(eval_dataloader)
        eval_acc_avg = eval_acc / total_num

        tqdm.write("\neval_loss: {}, eval_acc: {}".format(eval_loss_avg,
                                                          eval_acc_avg))

        self.__save_model()                                          

    def predict(self, sequence: List[List[np.ndarray]]) -> List[str]:
        """개채명 분류 모델 Inference 함수

        Args:
            sequence: 임베딩 문장
        
        Returns:
            List[str]: 추론 개채명 리스트
        """

        self.__load_model()

        pad_question = self.p.pad_question(sequence)
        pred_label = self.model(torch.tensor([pad_question]))[0]
        pred_class = []
        for idx in pred_label.max(dim=-1)[1].tolist():
            pred_class.append(self.model.label_dict[idx])
        return pred_class

    def __load_model(self):
        """학습 모델 호출"""

        if self.model_loaded is False:
            self.model_loaded = True

            state = torch.load(self.model_file + '.entity', 
                               map_location=torch.device('cpu'))
            self.model.load_state_dict(state['model_state_dict'])

    def __save_model(self):
        """학습 모델 저장"""

        state = {
            'label_dict': self.model.label_dict,
            'model_state_dict': self.model.to(torch.device('cpu')).state_dict()
        }

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(state, self.model_file + '.entity')
