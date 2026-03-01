import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from preprocessing import prepare_dataloaders
from model import BertAnxietyClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
BATCH_SIZE = 16
LR = 2e-5

def train():
    train_enc, val_enc, y_train, y_val = prepare_dataloaders(
        "final_anxiety_dataset.csv"
    )

    train_dataset = TensorDataset(
        train_enc['input_ids'],
        train_enc['attention_mask'],
        y_train
    )

    val_dataset = TensorDataset(
        val_enc['input_ids'],
        val_enc['attention_mask'],
        y_val
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BertAnxietyClassifier().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
                outputs = model(input_ids, attention_mask)
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true.extend(labels.cpu().numpy())

        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average="weighted")
        print(f"Val Accuracy: {acc:.4f} | Val F1: {f1:.4f}")

    torch.save(model.state_dict(), "../model/bert_anxiety_model.pt")
    print("Model saved.")

if __name__ == "__main__":
    train()
