import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os 


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )


    train_loss_history = []
    valid_loss_history = []


    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

            lr_scheduler.step()

        epoch_train_loss = avg_loss / batch_no
        train_loss_history.append(epoch_train_loss)

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            epoch_valid_loss = avg_loss_valid / batch_no
            valid_loss_history.append((epoch_no + 1, epoch_valid_loss))

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)

        loss_path = os.path.join(foldername, "loss_history.pk")
        with open(loss_path, "wb") as f:
            pickle.dump(
                {
                    "train": train_loss_history,
                    "valid": valid_loss_history,  
                },
                f,
            )
        print("✅ 已保存 loss 曲线到:", loss_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

#     with torch.no_grad():
#         model.eval()
#         mse_total = 0
#         mae_total = 0
#         evalpoints_total = 0

#         all_target = []
#         all_observed_point = []
#         all_observed_time = []
#         all_evalpoint = []
#         all_generated_samples = []
        
#         with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
#             for batch_no, test_batch in enumerate(it, start=1):
#                 output = model.evaluate(test_batch, nsample)

#                 samples, c_target, eval_points, observed_points, observed_time = output
#                 samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
#                 c_target = c_target.permute(0, 2, 1)  # (B,L,K)
#                 eval_points = eval_points.permute(0, 2, 1)
#                 observed_points = observed_points.permute(0, 2, 1)

#                 samples_median = samples.median(dim=1)
#                 all_target.append(c_target)
#                 all_evalpoint.append(eval_points)
#                 all_observed_point.append(observed_points)
#                 all_observed_time.append(observed_time)
#                 all_generated_samples.append(samples)

#                 mse_current = (
#                     ((samples_median.values - c_target) * eval_points) ** 2
#                 ) * (scaler ** 2)
#                 mae_current = (
#                     torch.abs((samples_median.values - c_target) * eval_points) 
#                 ) * scaler

#                 mse_total += mse_current.sum().item()
#                 mae_total += mae_current.sum().item()
#                 evalpoints_total += eval_points.sum().item()

#                 it.set_postfix(
#                     ordered_dict={
#                         "rmse_total": np.sqrt(mse_total / evalpoints_total),
#                         "mae_total": mae_total / evalpoints_total,
#                         "batch_no": batch_no,
#                     },
#                     refresh=True,
#                 )

#             with open(
#                 foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 all_target = torch.cat(all_target, dim=0)
#                 all_evalpoint = torch.cat(all_evalpoint, dim=0)
#                 all_observed_point = torch.cat(all_observed_point, dim=0)
#                 all_observed_time = torch.cat(all_observed_time, dim=0)
#                 all_generated_samples = torch.cat(all_generated_samples, dim=0)

#                 pickle.dump(
#                     [
#                         all_generated_samples,
#                         all_target,
#                         all_evalpoint,
#                         all_observed_point,
#                         all_observed_time,
#                         scalers,
#                         mean_scaler,
#                     ],
#                     f,
#                 )

#             CRPS = calc_quantile_CRPS(
#                 all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#             )
#             CRPS_sum = calc_quantile_CRPS_sum(
#                 all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#             )

#             with open(
#                 foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 pickle.dump(
#                     [
#                         np.sqrt(mse_total / evalpoints_total),
#                         mae_total / evalpoints_total,
#                         CRPS,
#                     ],
#                     f,
#                 )
#                 print("RMSE:", np.sqrt(mse_total / evalpoints_total))
#                 print("MAE:", mae_total / evalpoints_total)
#                 print("CRPS:", CRPS)
#                 print("CRPS_sum:", CRPS_sum)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", device = None):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        all_patient_ids = []   # 收集每个样本对应的 patient_id / 文件名

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # test_batch = move_to_device(test_batch, device)

                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)

                samples = torch.clamp(samples, -1.0, 1.0) # 限制生成值在合理范围内

                c_target = c_target.permute(0, 2, 1)   # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)

                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                # ===== ✅ 累积 patient_id（兼容 list / tensor / 标量）=====
                pid = test_batch.get("patient_id", None)
                if pid is not None:
                    if isinstance(pid, (list, tuple)):
                        all_patient_ids.extend([str(x) for x in pid])
                    elif isinstance(pid, torch.Tensor):
                        # 若是整型张量
                        all_patient_ids.extend([str(int(x)) for x in pid.view(-1)])
                    else:
                        all_patient_ids.append(str(pid))

                # ===== 统计 =====
                mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()
                evalpoints_total = max(evalpoints_total, 1) 

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total":  mae_total / evalpoints_total,
                        "batch_no":   batch_no,
                    },
                    refresh=True,
                )

            # ===== 保存 pk：在末尾加 patient_ids =====
            all_target = torch.cat(all_target, dim=0)
            all_evalpoint = torch.cat(all_evalpoint, dim=0)
            all_observed_point = torch.cat(all_observed_point, dim=0)
            all_observed_time = torch.cat(all_observed_time, dim=0)
            all_generated_samples = torch.cat(all_generated_samples, dim=0)

            save_path = os.path.join(foldername, f"generated_outputs_nsample{nsample}.pk")
            with open(save_path, "wb") as f:
                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,           
                        all_evalpoint,          
                        all_observed_point,    
                        all_observed_time,      
                        scaler,                
                        mean_scaler,           
                        all_patient_ids,        # patient_id 
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(os.path.join(foldername, f"result_nsample{nsample}.pk"), "wb") as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
                print("CRPS_sum:", CRPS_sum)


def move_to_device(batch, device):
    """递归地把 batch 中的 tensor 移到指定 device。
       特殊处理 patient_id，默认保持在 CPU，方便后续转成 str 保存。
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        new_batch = {}
        for k, v in batch.items():
            if k in ["patient_id", "pid", "subject_id"]:
                # 不移动 patient_id，保持在 CPU / 原始类型
                new_batch[k] = v
            else:
                new_batch[k] = move_to_device(v, device)
        return new_batch
    elif isinstance(batch, (list, tuple)):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch