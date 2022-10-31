import pytorch_lightning as pl
import torch
import torchmetrics
import transformers


class RegressionModel(pl.LightningModule):
    def __init__(self, model_name, lr, norm):
        '''
            # classification_loss
            1. L1Loss
            2. MSELoss
        '''
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.norm = norm

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1)
        
        # loss
        if self.norm > 1:
            self.distance_loss = torch.nn.MSELoss()
        else:
            self.distance_loss = torch.nn.SmoothL1Loss()  
        
    def forward(self, x):
        x = self.plm(x)['logits']
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        
        distance_loss = self.distance_loss(logits.squeeze(), labels.float())
        self.log("train_distance_loss", distance_loss)
        return distance_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        
        distance_loss = self.distance_loss(logits.squeeze(), labels.float())
        
        self.log("val_distance_loss", distance_loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.float()))
        return distance_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), labels.float()))

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
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.lr = lr
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        self.classification_loss = torch.nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        x = self.plm(x)['logits']
        return x    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        loss = self.classification_loss(logits.squeeze(), binary_labels.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        loss = self.classification_loss(logits.squeeze(), binary_labels.float())
        self.log("val_loss", loss)
        self.log("val_f1", torchmetrics.functional.f1_score(logits.squeeze(), binary_labels.int()))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        logits = self(x)
        self.log("test_f1", torchmetrics.functional.f1_score(logits.squeeze(), binary_labels.int()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
        return optimizer    
    

class EnsambleModel(pl.LightningModule):
    def __init__(self, 
            model_name,
            distance_model_path,        
            cls_model_path, 
            lr,
            norm,
        ):
        super(EnsambleModel, self).__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.norm = norm
        
        self.distance_model = RegressionModel(model_name, lr, norm).load_from_checkpoint(distance_model_path)
        self.cls_model = ClassificationModel(model_name, lr).load_from_checkpoint(cls_model_path)
        
        self.distance_model.freeze()
        self.cls_model.freeze()
        
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(1, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )
        self.fc_layers.apply(self.weight_initialization)
        
        if norm > 1:
            self.regression_loss = torch.nn.MSELoss()
        else:
            self.regression_loss = torch.nn.SmoothL1Loss()
        self.classification_loss = torch.nn.BCEWithLogitsLoss()
    
    def weight_initialization(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x):
        distance_model_out = self.distance_model.plm(x)['logits']
        cls_model_out = self.cls_model.plm(x)['logits']
        
        return {
            'dist_model_out': distance_model_out,
            'cls_model_out': cls_model_out
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        out = self(x)
        d_model_logits, c_model_logits = out['dist_model_out'], out['cls_model_out']
        sum_logits = d_model_logits + c_model_logits
    
        out = self.fc_layers(sum_logits)  # (b, 1)
        
        loss = self.regression_loss(out.squeeze(), labels.float())
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        out = self(x)
        d_model_logits, c_model_logits = out['dist_model_out'], out['cls_model_out']
        sum_logits = d_model_logits + c_model_logits
        out = self.fc_layers(sum_logits)  # (b, 1)
        
        loss = self.regression_loss(out.squeeze(), labels.float())
        self.log('val_loss', loss)
        self.log('val_pearson', torchmetrics.functional.pearson_corrcoef(out.squeeze(), labels.float()))
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        labels, binary_labels = y[:, 0], y[:, 1]
        
        out = self(x)
        d_model_logits, c_model_logits = out['dist_model_out'], out['cls_model_out']
        sum_logits = d_model_logits + c_model_logits
        out = self.fc_layers(sum_logits)  # (b, f)
                
        self.log('val_pearson', torchmetrics.functional.pearson_corrcoef(out.squeeze(), labels.float()))
        return
    
    def predict_step(self, batch, batch_idx):
        x = batch
        
        out = self(x)
        d_model_logits, c_model_logits = out['dist_model_out'], out['cls_model_out']
        sum_logits = d_model_logits + c_model_logits
        out = self.fc_layers(sum_logits)  # (b, f)        
        return out.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    