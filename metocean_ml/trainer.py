import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Trainer():
    """
    A general trainer for PyTorch models, designed to manage the training loop, 
    including loss computation, early stopping, learning rate reduction, and more.

    Parameters
    ----------
    train_lossfunc : torch.nn.Module, optional
        The loss function used for training. Default is `torch.nn.L1Loss()`.
    val_lossfunc : torch.nn.Module, optional
        The loss function used for validation. Default is `torch.nn.L1Loss()`.
    early_stopping_patience : int, optional
        The number of epochs with no improvement on validation loss after which
        training will be stopped. Default is 10.
    reduce_lr_patience : int, optional
        The number of epochs with no improvement on validation loss after which
        the learning rate will be reduced. Default is 5.
    minimum_lr : float, optional
        The minimum learning rate to which the learning rate can be reduced. Default is `1e-8`.
    improvement_threshold : float, optional
        The threshold for considering the validation loss to have improved. Default is `1e-7`.
    max_epochs : int, optional
        The maximum number of epochs to train the model. Default is 100.
    learning_rate : float, optional
        The initial learning rate. Default is `0.0001`.
    batch_size : int, optional
        The batch size for training. Default is 1024 (2^10).
    verbose : bool, optional
        If True, provides detailed logging during training. Default is True.

    """
    def __init__(
            self,
            train_lossfunc:nn.Module = nn.L1Loss(),
            val_lossfunc = nn.L1Loss(),
            early_stopping_patience = 10,
            reduce_lr_patience = 5,
            minimum_lr = 1e-8,
            improvement_threshold = 1e-7,
            max_epochs = 100,
            learning_rate = 0.0001,
            batch_size = 2**10,
            verbose=True,
            ):
        self.train_lossfunc = train_lossfunc
        self.val_lossfunc = val_lossfunc
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.minimum_lr = minimum_lr
        self.improvement_threshold = improvement_threshold
        self.max_epochs = max_epochs
        self.learning_rate_init = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
    
    def fit(self,
            model:nn.Module,
            train_data:DataLoader|Dataset,
            val_data:DataLoader|Dataset):
        """
        Trains the provided PyTorch model using the given training and validation datasets. 
        The method handles the training loop, validation, loss computation, early stopping, 
        and learning rate adjustments.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be trained.
        train_data : torch.utils.data.DataLoader or torch.utils.data.Dataset
            The training dataset. If a `Dataset` is provided, it will be wrapped in a `DataLoader`.
        val_data : torch.utils.data.DataLoader or torch.utils.data.Dataset
            The validation dataset. If a `Dataset` is provided, it will be wrapped in a `DataLoader`.

        Returns
        -------
        logs : dict
            A dictionary containing logs of train_loss, val_loss, learning_rate.
    
        """
        train_loss_log = []
        val_loss_log = []
        lr_log = []
        best_loss = float("inf")
        patience_counter = 0
        optimizer = torch.optim.Adam(model.parameters())
        learning_rate = self.learning_rate_init
        
        if not isinstance(train_data,DataLoader):
            train_data = DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
        if not isinstance(val_data,DataLoader):
            val_data = DataLoader(val_data,batch_size=self.batch_size,shuffle=False)
        
        pbar = tqdm(range(self.max_epochs),smoothing=0)
        for epoch in pbar:
            lr_log.append(learning_rate)

            # Training loop
            model.train()
            train_loss = 0.0
            for inputs, targets in train_data:
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = self.train_lossfunc(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_data.dataset)
            train_loss_log.append(train_loss)
            #print(loss,hs_loss)
            
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_data:
                    outputs = model(inputs).squeeze()
                    loss = self.val_lossfunc(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_data.dataset)
            val_loss_log.append(val_loss)
            
            # Info
            pbar.set_description('Epoch {}, Training loss: {}, validation loss: {}.'.format(epoch+1,round(train_loss,5),round(val_loss,5)))

            # Check if performance increased
            if val_loss + self.improvement_threshold < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Reduce lr or stop
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print("Early stopping triggered")
                break
            if patience_counter >= self.reduce_lr_patience:
                learning_rate *= 0.1
                if learning_rate <= self.minimum_lr:
                    break
                patience_counter = 0
                for g in optimizer.param_groups:
                    g['lr'] = learning_rate
                if self.verbose:
                    print(f"No improvement for {self.reduce_lr_patience} epochs. Lr reduced to {learning_rate}")

        return {"val_loss":val_loss_log,
                "train_loss":train_loss_log,
                "learning_rate":lr_log}

    def test(self,
             model:nn.Module,
             test_data:Dataset|DataLoader):
        """
        Evaluates the provided PyTorch model on the test dataset.
        The loss is computed using the validation loss function (`val_lossfunc`) specified during initialization.

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be evaluated.
        test_data : torch.utils.data.DataLoader or torch.utils.data.Dataset
            The test dataset. If a `Dataset` is provided, it will be wrapped in a `DataLoader`.

        Returns
        -------
        test_loss : float
            The average test loss across all batches in the test dataset.

        """
        if isinstance(test_data,Dataset):
            test_data = DataLoader(test_data,batch_size=self.batch_size,shuffle=False)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs,targets in test_data:
                outputs = model(inputs).squeeze()
                loss = self.val_lossfunc(outputs,targets)
                test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_data.dataset)