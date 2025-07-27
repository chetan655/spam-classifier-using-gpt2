import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")

# def text_to_ids(text, tokenizer):
#     token_ids = 


def calc_accuracy(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]   # logits of the last output token bcz last token have the attention score of all the other tokens
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples



def calc_batch_loss(input, target, model, device):
    input, target = input.to(device), target.to(device)
    output = model(input)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(output, target)
    return loss

def calc_loader_loss(data_loader, model, device, batch_size=None):
    total_loss = 0

    if len(data_loader) == 0:
        return torch.float('nan')
    
    if batch_size is None:
        batch_size = len(data_loader)
    else:
        batch_size = min(batch_size, len(data_loader))

    for i, (input, target) in enumerate(data_loader):
        if i < batch_size:
            loss = calc_batch_loss(input, target, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / batch_size