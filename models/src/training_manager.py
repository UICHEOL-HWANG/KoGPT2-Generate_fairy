from .model_manager import ModelManager
import torch
import numpy as np
from tqdm import tqdm 


class Training_Manager:
    def __init__(self, model, train_loader, val_loader):
        """
        Training Manager for KoGPT2 Fine-tuning

        Args:
            model: PyTorch model object (GPT2LMHeadModel)
            train_loader: PyTorch DataLoader for training
            val_loader: PyTorch DataLoader for validation
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = ModelManager().criterion  # Loss Function
        self.optimizer = ModelManager().optimizer  # Optimizer

    def train(self, epochs):
        """
        Train the model

        Args:
            epochs (int): Number of training epochs

        Returns:
            list: List of average losses per epoch
        """
        epoch_losses = []
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n🔄 Epoch [{epoch + 1}/{epochs}] 시작 - Learning Rate: {current_lr:.6e}")

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress.set_postfix(loss=loss.item())

            avg_loss = epoch_loss / len(self.train_loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        return epoch_losses

    def validation(self):
        """
        Validate the model

        Returns:
            tuple: (average validation loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            progress = tqdm(self.val_loader, desc="Validation", unit="batch")
            
            for batch in progress:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
                progress.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_val_loss, accuracy
