import torch
import tiktoken
from pathlib import Path
from tqdm import tqdm

from utils import get_model_weights_file, calc_batch_loss, calc_loader_loss, evaluate_model, calc_accuracy


def train(new_config, model, train_loader, val_loader, eval_freq, eval_iter):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)

    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Using gpt2 tokenizer...")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=new_config["learning_rate"], weight_decay=new_config["weight_decay"])

    global_step, initial_epoch = -1, 0

    Path(new_config["weight_folder"]).mkdir(parents=True, exist_ok=True)

    if new_config["preload"]:
        weight_filename = get_model_weights_file(new_config, new_config["preload"])
        print(f"Pre-loading weights -> {weight_filename}")

        state = torch.load(weight_filename)
        model.load_state_dict(state["model_state_dict"])

        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No weights found.")

    for epoch in range(initial_epoch, new_config["epoch"]):
        batch_iterator = tqdm(train_loader, desc=f"processing batch: {epoch:02d}")

        for input_batch, target_batch in batch_iterator:
            model.train()
            optimizer.zero_grad()
            loss = calc_batch_loss(input=input_batch, target=target_batch, device=device, model=model)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(train_loader=train_loader, val_loader=val_loader, device=device, model=model)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
            
                train_acc = calc_accuracy(data_loader=train_loader, device=device, model=model, num_batches=eval_iter)
                val_acc = calc_accuracy(data_loader=val_loader, model=model, device=device, num_batches=eval_iter)
            
                print(f"Training accuracy: {train_acc*100:.2f}%")
                print(f"validation accuracy: {val_acc*100:.2f}%")

        weight_filename = get_model_weights_file(new_config, epoch)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, weight_filename)

        print("weights and optimizer saved.")


