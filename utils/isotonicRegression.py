# Calculate isotonic regression loss
import torch
from torch import nn


def isotonic_regression_loss(args,outputs, seq_series):
    mse = nn.MSELoss(reduction='sum')
    # isotonic regression loss  for the generated full series
    B, T = seq_series.size()
    y_series_all = torch.arange(T).unsqueeze(dim=0).repeat(B, 1)
    y_series_all = (y_series_all / torch.max(y_series_all, dim=-1, keepdim=True)[0]).type(
        torch.float).to(outputs.device)  # normalization

    seq_pred_series = seq_series / torch.max(seq_series, dim=-1, keepdim=True)[0].detach()

    isot_regres_loss_all = mse(seq_pred_series, y_series_all)

    # isotonic regression loss for the input series
    y_series_input = torch.arange(args.seq_len).unsqueeze(dim=0).repeat(B, 1)
    y_series_input = (y_series_input / torch.max(y_series_input, dim=-1, keepdim=True)[0]).type(
        torch.float).to(outputs.device)  # normalization

    seq_series_input = seq_series[:, :args.seq_len] / \
                       torch.max(seq_series[:, :args.seq_len], dim=-1, keepdim=True)[0].detach()

    isot_regres_loss_input = mse(y_series_input, seq_series_input)

    # isotonic regression loss for the prediction series
    y_series_pred = torch.arange(args.pred_len).unsqueeze(dim=0).repeat(B, 1)
    y_series_pred = (y_series_pred / torch.max(y_series_pred, dim=-1, keepdim=True)[0]).type(
        torch.float).to(outputs.device)

    seq_series_pred = seq_series[:, -args.pred_len:] / \
                      torch.max(seq_series[:, -args.pred_len:], dim=-1, keepdim=True)[0].detach()

    isot_regres_loss_pred = mse(y_series_pred, seq_series_pred)

    return isot_regres_loss_all, isot_regres_loss_input, isot_regres_loss_pred


# Calculating the isotonic regression loss using sliding windows
def isotonic_regression_loss_sliding_window(args,win_len, stride, outputs, seq_series):
    mse = nn.MSELoss(reduction='sum')
    B, T = seq_series.size()
    t = 0
    loss_all = 0
    # isotonic regression loss  for the generated full series
    for i in range(0, T + 1 - win_len, stride):
        t += 1
        seq_all_part = seq_series[:, i:(i + win_len)] / torch.max(seq_series[:, i:(i + win_len)], dim=-1, keepdim=True)[
            0].detach()
        y_series_part = torch.arange(win_len).unsqueeze(dim=0).repeat(B, 1)
        y_series_part_ = (y_series_part / torch.max(y_series_part, dim=-1, keepdim=True)[0]).type(
            torch.float).to(outputs.device)
        loss_part = mse(seq_all_part, y_series_part_)
        loss_all = loss_all + loss_part
    loss_all = loss_all / t

    # isotonic regression loss for the input series
    h = 0
    loss_raw = 0
    seq_series_raw = seq_series[:, :args.seq_len]
    for i in range(0, args.seq_len + 1 - win_len, stride):
        h = h + 1
        seq_series_input = seq_series_raw[:, i:(i + win_len)] / \
                           torch.max(seq_series_raw[:, i:(i + win_len)], dim=-1, keepdim=True)[0].detach()

        y_series_input = torch.arange(win_len).unsqueeze(dim=0).repeat(B, 1)
        y_series_input_ = (y_series_input / torch.max(y_series_input, dim=-1, keepdim=True)[0]).type(torch.float).to(
            outputs.device)

        loss_raw_part = mse(seq_series_input, y_series_input_)
        loss_raw = loss_raw + loss_raw_part
    loss_raw = loss_raw / h

    # isotonic regression loss for the prediction series
    k = 0
    loss_pred = 0
    seq_series_pred = seq_series[:, -args.pred_len:]
    for i in range(0, args.pred_len + 1 - win_len, stride):
        k = k + 1
        seq_series_output = seq_series_pred[:, i:(i + win_len)] / \
                            torch.max(seq_series_pred[:, i:(i + win_len)], dim=-1, keepdim=True)[0].detach()

        y_series_pred = torch.arange(win_len).unsqueeze(dim=0).repeat(B, 1)
        y_series_pred_ = (y_series_pred / torch.max(y_series_pred, dim=-1, keepdim=True)[0]).type(
            torch.float).to(outputs.device)

        loss_pred_part = mse(seq_series_output, y_series_pred_)
        loss_pred = loss_pred + loss_pred_part
    loss_pred = loss_pred / k

    return loss_all, loss_raw, loss_pred


def isotonic_regression_loss_impt(args,outputs, seq_series):
    mse = nn.MSELoss(reduction='sum')
    # isotonic regression loss  for the generated full series
    B, T = seq_series.size()
    y_series_all = torch.arange(T).unsqueeze(dim=0).repeat(B, 1)
    y_series_all = (y_series_all / torch.max(y_series_all, dim=-1, keepdim=True)[0]).type(
        torch.float).to(outputs.device)  # normalization

    seq_pred_series = seq_series / torch.max(seq_series, dim=-1, keepdim=True)[0].detach()

    isot_regres_loss_all = mse(seq_pred_series, y_series_all)

    # isotonic regression loss for the input series
    y_series_input = torch.arange(args.seq_len).unsqueeze(dim=0).repeat(B, 1)
    y_series_input = (y_series_input / torch.max(y_series_input, dim=-1, keepdim=True)[0]).type(
        torch.float).to(outputs.device)  # normalization

    seq_series_input = seq_series[:, :args.seq_len] / \
                       torch.max(seq_series[:, :args.seq_len], dim=-1, keepdim=True)[0].detach()

    isot_regres_loss_input = mse(y_series_input, seq_series_input)

    return isot_regres_loss_all, isot_regres_loss_input


# Calculating the isotonic regression loss using sliding windows
def isotonic_regression_loss_sliding_window_impt(args,win_len, stride, outputs, seq_series):
    mse = nn.MSELoss(reduction='sum')
    B, T = seq_series.size()
    t = 0
    loss_all = 0
    # isotonic regression loss  for the generated full series
    for i in range(0, T + 1 - win_len, stride):
        t += 1
        seq_all_part = seq_series[:, i:(i + win_len)] / torch.max(seq_series[:, i:(i + win_len)], dim=-1, keepdim=True)[
            0].detach()
        y_series_part = torch.arange(win_len).unsqueeze(dim=0).repeat(B, 1)
        y_series_part_ = (y_series_part / torch.max(y_series_part, dim=-1, keepdim=True)[0]).type(
            torch.float).to(outputs.device)
        loss_part = mse(seq_all_part, y_series_part_)
        loss_all = loss_all + loss_part
    loss_all = loss_all / t

    # isotonic regression loss for the input series
    h = 0
    loss_raw = 0
    seq_series_raw = seq_series[:, :args.seq_len]
    for i in range(0, args.seq_len + 1 - win_len, stride):
        h = h + 1
        seq_series_input = seq_series_raw[:, i:(i + win_len)] / \
                           torch.max(seq_series_raw[:, i:(i + win_len)], dim=-1, keepdim=True)[0].detach()

        y_series_input = torch.arange(win_len).unsqueeze(dim=0).repeat(B, 1)
        y_series_input_ = (y_series_input / torch.max(y_series_input, dim=-1, keepdim=True)[0]).type(torch.float).to(
            outputs.device)

        loss_raw_part = mse(seq_series_input, y_series_input_)
        loss_raw = loss_raw + loss_raw_part
    loss_raw = loss_raw / h

    return loss_all, loss_raw