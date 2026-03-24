import torch
# 损失函数
def mil_loss(y_pred, y_true, A, H,
             lambda_t=0.1,
             lambda_s=0.5,
             lambda_i=0.1):
    y_true = y_true.view_as(y_pred)
    # ===== BCE（带focal）
    gamma = 2.0
    y_pred = torch.clamp(y_pred, 1e-6, 1-1e-6)

    bce = - (
        y_true * (1 - y_pred) ** gamma * torch.log(y_pred) +
        (1 - y_true) * y_pred ** gamma * torch.log(1 - y_pred)
    ).mean()

    # ===== 时间平滑
    temp = torch.mean((A[:,1:] - A[:,:-1])**2)

    # ===== 稀疏性
    sparsity = torch.mean(torch.abs(A))

    # ===== instance一致性
    B, T, C = H.shape
    H1 = H.unsqueeze(2)
    H2 = H.unsqueeze(1)
    inst = torch.mean((H1 - H2)**2)

    loss = bce + lambda_t*temp + lambda_s*sparsity + lambda_i*inst

    return loss