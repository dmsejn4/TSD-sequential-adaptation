#1.Fisher 계산
fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
for batch in task1_loader:            # Task 1 train split 일부 (예: 500~1000장)
    model.zero_grad()
    loss, _ = model.loss(batch)
    loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            fisher[n] += p.grad.detach() ** 2 / len(task1_loader)

#2.학습 루프에 추가할 penalty def
def ewc_penalty(model, fisher, theta_star, lam):
    pen = 0.0

    for n, p in model.named_parameters():

        if n not in fisher:
            continue

        if n not in theta_star:
            continue

        if p.shape != fisher[n].shape:
            continue

        if p.shape != theta_star[n].shape:
            continue

        pen += (
            fisher[n].to(p.device)
            * (p - theta_star[n].to(p.device)) ** 2
        ).sum()

    return lam / 2 * pen.float()
