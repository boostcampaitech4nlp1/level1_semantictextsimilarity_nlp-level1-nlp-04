import pytorch_lightning as pl
import torch
import torchmetrics
import transformers


class RegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, norm, cls_weight):
        '''
            # criterion
            1. L1Loss
            2. MSELoss
        '''
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.norm = norm
        self.cls_weight = cls_weight

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        
        # loss
        if self.norm > 1:
            self.distance_loss = torch.nn.MSELoss()
        else:
            self.distance_loss = torch.nn.L1Loss()  
        self.classification_loss = torch.nn.BCEWithLogitsLoss()
        
        self.w = torch.nn.Parameter(torch.FloatTensor([0.9]))

    def forward(self, x):
        x = self.plm(x)['logits']
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        distance_loss = self.distance_loss(logits.squeeze(), labels.float())
        classification_loss = self.classification_loss(logits.squeeze(), binary_labels.float())
        total_loss = distance_loss + self.cls_weight * classification_loss
        
        self.log("train_distance_loss", distance_loss)
        self.log("train_classification_loss", classification_loss)
        self.log("train_total_loss", total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        distance_loss = self.distance_loss(logits.squeeze(), labels.float())
        classification_loss = self.classification_loss(logits.squeeze(), binary_labels.float())
        total_loss = distance_loss + self.cls_weight * classification_loss
        
        self.log("val_total_loss", total_loss)
        
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.float()))
        self.log("val_f1", torchmetrics.functional.f1_score(logits.squeeze(), binary_labels.float()))
        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.float()))
        self.log("test_f1", torchmetrics.functional.f1_score(logits.squeeze(), binary_labels.float()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
        return optimizer
    
class ClassificationModel(pl.LightningModule):
    def __init__(self, model_name, lr):
        super(ClassificationModel, self).__init__()
        
        self.model_name = model_name
        self.lr = lr
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        loss = self.criterion(logits, binary_labels.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        loss = self.criterion(logits, binary_labels.float())
        self.log("val_loss", loss)
        self.log("val_f1", torchmetrics.functional.f1_score(logits.squeeze(), binary_labels))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        self.log("test_pearson", torchmetrics.functional.f1_score(logits.squeeze(), binary_labels))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
        return optimizer    
        
    