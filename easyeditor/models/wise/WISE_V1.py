import copy
import os
import random
import pickle
import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc
import json
import matplotlib.pyplot as plt
import pickle  # %wys

merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []


def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance

    act_fn = ACT2FN[config.hidden_act]
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if act_mask is not None:
        return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)


class WISE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(WISE, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None

        self.loss_meter = EarlyStopMeter()  # wys
        self.teacher_logits = None  # 初始化教师logits（兼容旧字段）
        self.is_first_in_group = False  # 组内第一个样本标记
        self.final_first_sample_logits = None#wys
        #self.group_teacher_logits = None  # 组内教师logits缓存（替代磁盘temp.pkl）
        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add WISE to chosen layers ---
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        if type(adapter_layer) is not WISEAdapter:
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, adapter_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, **kwargs):
        if not self.config.retrieve:
            if hasattr(self.get_adapter_layer(), 'editing') and not self.get_adapter_layer().editing:
                # final merge
                if not self.get_adapter_layer().original_layer.weight.equal(
                        self.get_adapter_layer().new_weight) and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    self.get_adapter_layer().memory_weight.append(self.get_adapter_layer().new_weight)
                if len(self.get_adapter_layer().memory_weight) > 0 and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    print('length of memory is ', len(self.get_adapter_layer().memory_weight), '!!!!!!')
                    self.get_adapter_layer().merge_weight()
        return self.model(**kwargs)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is WISEAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer.to(self.model.device)

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}": v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1
        # input_ids  labels

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)
            # self.get_adapter_layer().generate_fixed_mask(self.config.mask_line) #wys
        print(f'self.config.distill=={self.config.distill}')
        # --- train Wise value ---
        if not self.config.distill:
            loss_meter = EarlyStopMeter()
            for i in range(config.n_iter):

                if i == 0:
                    # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                    optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr,
                                                weight_decay=1e-5)

                ft_loss, _ = self._cal_ft_loss(tokens, last_prompt_token_loc)

                act_loss = self._cal_activation_loss(self.get_adapter_layer().original_layer_output,
                                                     self.get_adapter_layer().new_weight_layer_output,
                                                     config=config, act_mask=act_mask, deact_mask=deact_mask)
                loss = ft_loss + act_loss.to(ft_loss.device)

                if loss_meter.stop():
                    self.get_adapter_layer().save_editing_activation()  # add last gradient
                    break
                if i == config.n_iter - 1:
                    self.get_adapter_layer().save_editing_activation()  # add last gradient

                if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                    memory_loss = []
                    for _ in merge_group_edit_history:
                        idx = 0
                        while True:
                            memo_input, is_used = _[idx]
                            if not is_used:
                                _[idx][1] = True
                                break
                            idx += 1
                            if idx == len(_):  ## re Assign
                                for m in range(len(_)):
                                    _[m][1] = False
                                idx = 0

                        memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                        self.model(**memo_input)

                        memory_act_loss = self._cal_memory_neg_activation_loss(
                            self.get_adapter_layer().original_layer_output,
                            self.get_adapter_layer().new_weight_layer_output, config=config,
                            act_mask=act_mask, deact_mask=deact_mask)
                        memory_loss.append(memory_act_loss.to(ft_loss.device))
                        del memo_input
                    neg_memo_loss = torch.stack(memory_loss).mean()
                    loss += neg_memo_loss
                    if len(edit_history) > 0:
                        memo_input = random.choice(edit_history)[0]
                        memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                        self.model(**memo_input)

                        pos_memo_loss = self._cal_memory_pos_activation_loss(
                            self.get_adapter_layer().original_layer_output,
                            self.get_adapter_layer().new_weight_layer_output, config=config,
                            act_mask=act_mask, deact_mask=deact_mask)
                        del memo_input
                        loss += pos_memo_loss.to(ft_loss.device)
                # for replay Appendix B.3

                optimizer.zero_grad()

                loss.backward()
                self.get_adapter_layer().mask_new_weight_gradient()

                if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                    print(
                        f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                    )
                else:
                    print(
                        f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                    )

                optimizer.step()
                loss_meter.update(loss.item())
                if type(self.config.norm_constraint) is float:
                    self._norm_constraint(self.config.norm_constraint)

        #     ##wys
        # 组内分布蒸馏：以 merge 组为单位（merge_freq 为组大小）
        else:
            if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") in [self.config.save_freq * 1,
                                                                                  self.config.save_freq * 2,
                                                                                  self.config.save_freq * 3]:
                print('getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt")',
                      getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt"))
                self.loss_meter = EarlyStopMeter()
                self.teacher_logits = None  # 初始化教师logits
                self.is_first_in_group = True  # 标记是否是组内的第一个样本
                # self.final_first_sample_logits = None  # 存储第一个样本训练完成后的logits
                a = None
                with open('temp.pkl', 'wb') as f:
                    pickle.dump(a, f)  # 移动到 CPU 再存储

            with open('temp.pkl', 'rb') as f:
                loaded_tensor = pickle.load(f)
            try:
                self.final_first_sample_logits = loaded_tensor.cuda()
            except:
                self.final_first_sample_logits = loaded_tensor
            for i in range(config.n_iter):
                if i == 0:
                    # 创建优化器（仅第一次迭代需要）
                    optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr,
                                                weight_decay=1e-5)

                # 计算微调损失并获取当前logits
                ft_loss, current_logits = self._cal_ft_loss(tokens, last_prompt_token_loc)

                # 初始化蒸馏损失
                distill_loss = 0.0

                # 如果不是第一个样本且有教师logits，计算蒸馏损失
                # print(self.is_first_in_group,self.final_first_sample_logits)
                if not self.is_first_in_group and self.final_first_sample_logits is not None:
                    T = config.temperature  # 温度系数
                    teacher_logits = self.final_first_sample_logits.unsqueeze(
                        0) if self.final_first_sample_logits.dim() == 1 else self.final_first_sample_logits
                    student_logits = current_logits.unsqueeze(0) if current_logits.dim() == 1 else current_logits

                    # 获取教师和学生logits的序列长度
                    teacher_seq_len = teacher_logits.shape[1]  # 序列长度维度
                    student_seq_len = student_logits.shape[1]

                    # 确定最大序列长度
                    max_seq_len = max(teacher_seq_len, student_seq_len)

                    # 创建填充函数
                    def pad_logits(logits, target_length):
                        """
                        将logits填充到目标长度

                        参数:
                            logits: 形状为 [batch_size, seq_len, vocab_size] 的张量
                            target_length: 目标序列长度

                        返回:
                            填充后的logits和对应的注意力掩码
                        """
                        batch_size, seq_len, vocab_size = logits.shape
                        if seq_len == target_length:
                            # 如果已经是对应长度，直接返回
                            return logits, torch.ones(batch_size, target_length, dtype=torch.bool, device=logits.device)

                        # 创建填充后的张量
                        padded_logits = torch.zeros(batch_size, target_length, vocab_size,
                                                    device=logits.device, dtype=logits.dtype)

                        # 创建注意力掩码（1表示真实token，0表示填充token）
                        attention_mask = torch.zeros(batch_size, target_length,
                                                     dtype=torch.bool, device=logits.device)

                        # 将原始logits复制到填充张量的开头
                        padded_logits[:, :seq_len, :] = logits
                        attention_mask[:, :seq_len] = True  # 标记真实token位置

                        return padded_logits, attention_mask

                    # 对教师和学生logits进行填充
                    teacher_padded, teacher_mask = pad_logits(teacher_logits, max_seq_len)
                    student_padded, student_mask = pad_logits(student_logits, max_seq_len)

                    # 计算KL散度损失（考虑填充）
                    loss_fn = torch.nn.KLDivLoss(reduction='none')
                    soft_target = F.softmax(teacher_padded / T, dim=-1)
                    output_log = F.log_softmax(student_padded / T, dim=-1)

                    # 计算每个位置的KL散度
                    kl_per_element = loss_fn(output_log, soft_target)

                    # 创建联合注意力掩码（只有教师和学生都是真实token的位置才计算损失）
                    joint_mask = teacher_mask.unsqueeze(-1) & student_mask.unsqueeze(-1)

                    # 应用掩码并计算平均损失
                    masked_kl = kl_per_element * joint_mask
                    num_valid_elements = joint_mask.sum()

                    if num_valid_elements > 0:
                        distill_loss = masked_kl.sum() / num_valid_elements * (T ** 2) * config.lamda
                    else:
                        # 如果没有有效元素，返回零损失
                        distill_loss = torch.tensor(0.0, device=teacher_logits.device)
                    # # 确保维度一致
                    # teacher_logits = self.final_first_sample_logits.unsqueeze(
                    #     0) if self.final_first_sample_logits.dim() == 1 else self.final_first_sample_logits
                    # student_logits = current_logits.unsqueeze(0) if current_logits.dim() == 1 else current_logits
                    #
                    # # 计算KL散度损失
                    # loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
                    # soft_target = F.softmax(teacher_logits / T, dim=-1)
                    # output_log = F.log_softmax(student_logits / T, dim=-1)
                    # distill_loss = loss_fn(output_log, soft_target) * (T ** 2) * config.lamda

                # 计算激活损失
                act_loss = self._cal_activation_loss(
                    self.get_adapter_layer().original_layer_output,
                    self.get_adapter_layer().new_weight_layer_output,
                    config=config, act_mask=act_mask, deact_mask=deact_mask
                )

                # 总损失（微调损失 + 激活损失 + 蒸馏损失）
                loss = ft_loss + act_loss.to(ft_loss.device) + distill_loss

                # 检查是否需要提前停止
                print(f'索引值：{i}')
                if self.loss_meter.stop():
                    self.get_adapter_layer().save_editing_activation()  # add last gradient
                    # 如果是第一个样本，保存最终logits作为教师logits
                    if self.is_first_in_group:
                        print('tag1')
                        self.final_first_sample_logits = current_logits  # .detach().clone()
                    break

                if i == config.n_iter - 1:
                    self.get_adapter_layer().save_editing_activation()  # add last gradient
                    # 如果是第一个样本，保存最终logits作为教师logits
                    if self.is_first_in_group:
                        print('tag2')
                        with open('temp.pkl', 'wb') as f:
                            pickle.dump(current_logits.cpu(), f)  # 移动到 CPU 再存储
                        # self.final_first_sample_logits = current_logits#.detach().clone()

                # 处理记忆回放（如果启用）
                if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                    memory_loss = []
                    for _ in merge_group_edit_history:
                        idx = 0
                        while True:
                            memo_input, is_used = _[idx]
                            if not is_used:
                                _[idx][1] = True
                                break
                            idx += 1
                            if idx == len(_):  ## re Assign
                                for m in range(len(_)):
                                    _[m][1] = False
                                idx = 0

                        memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                        self.model(**memo_input)

                        memory_act_loss = self._cal_memory_neg_activation_loss(
                            self.get_adapter_layer().original_layer_output,
                            self.get_adapter_layer().new_weight_layer_output, config=config,
                            act_mask=act_mask, deact_mask=deact_mask)
                        memory_loss.append(memory_act_loss.to(ft_loss.device))
                        del memo_input
                    neg_memo_loss = torch.stack(memory_loss).mean()
                    loss += neg_memo_loss
                    if len(edit_history) > 0:
                        memo_input = random.choice(edit_history)[0]
                        memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                        self.model(**memo_input)

                        pos_memo_loss = self._cal_memory_pos_activation_loss(
                            self.get_adapter_layer().original_layer_output,
                            self.get_adapter_layer().new_weight_layer_output, config=config,
                            act_mask=act_mask, deact_mask=deact_mask)
                        del memo_input
                        loss += pos_memo_loss.to(ft_loss.device)

                # 梯度清零
                optimizer.zero_grad()

                # 反向传播
                loss.backward()
                self.get_adapter_layer().mask_new_weight_gradient()

                # 打印损失信息
                if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                    print(
                        f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}+{distill_loss}"
                    )
                else:
                    print(
                        f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}+{distill_loss}"
                    )

                # 优化步骤
                optimizer.step()
                self.loss_meter.update(loss.item())

                # 在第一个样本处理完成后更新标志
                if i == config.n_iter - 1 or self.loss_meter.stop():
                    if self.is_first_in_group:
                        self.is_first_in_group = False  # 标记第一个样本处理完成

                #     ## wys end

                if type(self.config.norm_constraint) is float:
                    self._norm_constraint(self.config.norm_constraint)

        plt.savefig('training_curves_only.png', dpi=300, bbox_inches='tight')
        plt.show()
        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)

        # 新增：当完成merge_cnt个编辑后，保存当前批次的侧记忆池ID记录
        if editing_total_cnt % self.config.merge_freq == 0:
            if len(self.get_adapter_layer().current_edit_batch) > 0:
                self.get_adapter_layer().edit_memory_records.append(
                    copy.deepcopy(self.get_adapter_layer().current_edit_batch)
                )
                print(f'保存了{len(self.get_adapter_layer().current_edit_batch)}个编辑的侧记忆池ID记录')
                self.get_adapter_layer().current_edit_batch = []  # 清空当前批次记录

        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            self.get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_weight()
            # 完成一个merge组后，清空教师logits缓存
            self.group_teacher_logits = None
            ##wys
            # if editing_total_cnt==20:
            #     self.get_adapter_layer().memory_weight.pop(-1)
            #     self.get_adapter_layer().memory_weight.append(copy.deepcopy(self.get_adapter_layer().original_layer.weight))

            # print(f'editing_total_cnt===={editing_total_cnt}')
            # if editing_total_cnt>30:
            # 保存pkl
            # data=self.get_adapter_layer().memory_weight
            # with open(f"memory_weight.pkl", "wb") as f:  # 注意模式必须是二进制写入 'wb'
            #     pickle.dump(data, f)  # 默认协议为最高版本（Python 3 推荐）
            # print("memory weight successfully saved")
            # # 保存memory_mean_act
            # data2=self.get_adapter_layer().memory_mean_act
            # with open(f"memory_mean_act.pkl", "wb") as f:  # 注意模式必须是二进制写入 'wb'
            #     pickle.dump(data2, f)  # 默认协议为最高版本（Python 3 推荐）
            # #读取pkl
            # with open("memory_weight.pkl", "rb") as f:  # 注意必须是二进制读取模式 'rb'
            #     memory_weight = pickle.load(f)
            # with open("memory_mean_act.pkl", "rb") as f:  # 注意必须是二进制读取模式 'rb'
            #     memory_mean_act = pickle.load(f)
            # for i,j in enumerate(memory_weight):
            #     # self.get_adapter_layer().memory_weight.append(j)
            #     # self.get_adapter_layer().memory_mean_act.append(memory_mean_act[i])
            #
            #     self.get_adapter_layer().add_elements(j,memory_mean_act[i])

            # min_a = 1e9
            # for _ in range(len(memory_weight)):
            #     memory_weight.pop()
            #     # edit_act = self.memory_mean_act.pop()
            #     # min_a = min(min_a, edit_act.min_act())
            #     self.get_adapter_layer().memory_mean_act.append(EditingMeanAct(min_a=min_a))
            # print('memory_weight merged successfully')
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _norm_constraint(self, norm_constraint):
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def _cal_ft_loss(self, tokens, last_prompt_token_loc):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits

        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bs, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1:] = True

        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss, logits  # wys

    def _cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                             deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        len_temp = original_layer_output.shape[0] / k - 1
        for i, act_mk in enumerate(act_mask):
            if act_mk is not None:
                in_scope_dist = euc(original_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...],
                                    new_weight_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...], config,
                                    act_mask=act_mk)
                out_scope_dist = euc(original_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...],
                                     new_weight_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...], config,
                                     act_mask=deact_mask[i])
            else:
                in_scope_dist = euc(original_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...],
                                    new_weight_layer_output[int(i * len_temp):int((i + 1) * len_temp), ...], config)
                if (i == k - 1):
                    out_scope_dist = euc(original_layer_output[int(i - k):, ...],
                                         new_weight_layer_output[int(i - k):, ...], config)
                else:
                    out_scope_dist = euc(original_layer_output[int(i - k):int(i + 1 - k), ...],
                                         new_weight_layer_output[int(i - k):int(i + 1 - k), ...], config)

            loss = out_scope_dist.view(-1, 1) - in_scope_dist + config.gamma
            loss2 = out_scope_dist - config.alpha
            loss3 = config.beta - in_scope_dist
            loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(
                original_layer_output.device)
            loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(
                original_layer_output.device)
            loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(
                original_layer_output.device)
            total_loss.append(loss + loss2 + loss3)
        return sum(total_loss) / len(total_loss)

    def _cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None,
                                        act_mask=None,
                                        deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def _cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None,
                                        act_mask=None,
                                        deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def save(self, save_path):
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save additional information, such as memory_weight, memory_mean_act, etc.
        additional_info = {
            'memory_weight': self.get_adapter_layer().memory_weight,
            'memory_mean_act': self.get_adapter_layer().memory_mean_act,
            'merge_cnt': self.get_adapter_layer().merge_cnt,
            'editing_mean_act': self.get_adapter_layer().editing_mean_act,
            'editing_total_cnt': self.get_adapter_layer().editing_total_cnt,
            'weight_mask': self.get_adapter_layer().weight_mask,
            'edit_memory_records': self.get_adapter_layer().edit_memory_records,
            'current_edit_batch': self.get_adapter_layer().current_edit_batch,
            # Add other variables that need to be saved
        }
        if hasattr(self.get_adapter_layer(), 'key_id') and self.get_adapter_layer().key_id is not None:
            additional_info['key_id'] = self.get_adapter_layer().key_id
        # Save all information to the file
        torch.save({
            'adapter_state_dict': self.get_adapter_layer().state_dict(),
            'config': self.config,
            'additional_info': additional_info,
            'edit_history': edit_history,
            'merge_group_edit_history': merge_group_edit_history
        }, save_path)

    def load(self, load_path):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        # Load all previously saved information
        saved_data = torch.load(load_path)
        if hasattr(self.model.config, 'hidden_act'):
            saved_data['config'].hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            saved_data['config'].hidden_act = self.model.config.activation_function
        if saved_data['config'] != self.config:
            print("Warning: The loaded WISE config is different from the original config")

        # Restore the state dictionary of the WISE Adapter instance
        self.get_adapter_layer().load_state_dict(saved_data['adapter_state_dict'])
        # Restore additional information
        adapter_layer = self.get_adapter_layer()
        for key, value in saved_data['additional_info'].items():
            setattr(adapter_layer, key, value)

        # Restore editing history
        global edit_history, merge_group_edit_history
        edit_history = saved_data['edit_history']
        merge_group_edit_history = saved_data['merge_group_edit_history']
        print(f"Model configuration and WISE state loaded from {load_path}")

    def evaluate_rewrite_accuracy(self, test_samples):
        """
        评估重写准确率，识别错误样本

        Args:
            test_samples: 测试样本列表，每个样本包含input_ids和expected_output

        Returns:
            rewrite_acc: 重写准确率
            error_samples: 错误样本列表（rewrite_acc < 1的样本）
            error_memory_ids: 错误样本对应的侧记忆池ID列表
        """
        print("开始评估重写准确率...")

        # 确保模型处于评估模式
        self.model.eval()
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        correct_count = 0
        total_count = len(test_samples)
        error_samples = []
        error_memory_ids = []
        device = self.model.device
        # 获取当前批次的侧记忆池ID记录
        current_batch_memory_ids = self.get_adapter_layer().current_edit_batch
        print(f'test_samples={test_samples}')
        with torch.no_grad():
            for i, sample in enumerate(test_samples):
                # 准备输入
                input_ids = sample['input_ids']['input_ids']  # .to(self.device)
                attention_mask = sample['input_ids']['attention_mask']
                # %
                # #input_ids=np.array(input_ids)
                # expected_output = sample['expected_output']
                #
                # #生成输出
                # generated_output = self.model.generate(
                #     input_ids=input_ids,
                #     #max_length=input_ids.shape[1] + len(expected_output),
                #     do_sample=False,
                #     pad_token_id=self.model.config.eos_token_id if hasattr(self.model.config, 'eos_token_id') else 0
                # )

                if not torch.is_tensor(input_ids):
                    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
                else:
                    input_ids = input_ids.to(device)

                if not torch.is_tensor(attention_mask):
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
                else:
                    attention_mask = attention_mask.to(device)

                expected_output = sample['expected_output']['input_ids']  # List[int]
                gen_len = max(1, len(expected_output))  # 也可设固定值

                # 生成输出（用 **inputs 传入完整张量）
                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
                generated_output = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_len,  # 更推荐
                    do_sample=False,
                    pad_token_id=getattr(self.model.config, 'eos_token_id', 0)
                )

                # 提取生成的部分（去掉输入部分）
                generated_text = generated_output[0][input_ids.shape[1]:]
                expected_text = torch.tensor(expected_output).to(device)

                # 计算准确率（完全匹配）
                if generated_text.shape[0] >= expected_text.shape[0]:
                    generated_text = generated_text[:expected_text.shape[0]]
                    is_correct = torch.equal(generated_text, expected_text)
                else:
                    is_correct = False

                if is_correct:
                    correct_count += 1
                else:
                    error_samples.append(sample)
                    # 记录对应的侧记忆池ID（如果存在）
                    if i < len(current_batch_memory_ids):
                        error_memory_ids.append(current_batch_memory_ids[i])
                    else:
                        error_memory_ids.append(-1)  # 默认值

                print(f"样本 {i + 1}/{total_count}: {'正确' if is_correct else '错误'}")

        rewrite_acc = correct_count / total_count
        print(f"重写准确率: {rewrite_acc:.4f}")
        print(f"错误样本数量: {len(error_samples)}")

        return rewrite_acc, error_samples, error_memory_ids

    def find_most_error_prone_memory_pool(self, error_memory_ids):
        """
        统计错误样本最多的侧记忆池ID

        Args:
            error_memory_ids: 错误样本对应的侧记忆池ID列表

        Returns:
            most_error_memory_id: 错误样本最多的侧记忆池ID
            error_counts: 每个侧记忆池ID的错误次数统计
        """
        print("统计错误样本最多的侧记忆池ID...")

        # 统计每个侧记忆池ID的错误次数
        error_counts = {}
        for memory_id in error_memory_ids:
            if memory_id != -1 and memory_id != -2:  # 排除无效ID
                error_counts[memory_id] = error_counts.get(memory_id, 0) + 1

        if not error_counts:
            print("没有找到有效的侧记忆池ID错误记录")
            return -1, error_counts

        # 找到错误次数最多的侧记忆池ID
        most_error_memory_id = max(error_counts, key=error_counts.get)
        max_error_count = error_counts[most_error_memory_id]

        print(f"错误统计结果:")
        for memory_id, count in sorted(error_counts.items()):
            print(f"  侧记忆池ID {memory_id}: {count} 个错误样本")

        print(f"错误样本最多的侧记忆池ID: {most_error_memory_id} (错误次数: {max_error_count})")

        return most_error_memory_id, error_counts

    def reinitialize_memory_pool(self, memory_pool_id):
        """
        重新初始化指定的侧记忆池

        Args:
            memory_pool_id: 要重新初始化的侧记忆池ID
        """
        print(f"开始重新初始化侧记忆池ID: {memory_pool_id}")

        adapter_layer = self.get_adapter_layer()

        # 检查侧记忆池ID是否有效
        if memory_pool_id < 0 or memory_pool_id >= len(adapter_layer.memory_weight):
            print(f"无效的侧记忆池ID: {memory_pool_id}")
            return False

        # 重新初始化指定侧记忆池的权重
        with torch.no_grad():
            # 使用原始层权重作为基础进行重新初始化
            original_weight = adapter_layer.original_layer.weight

            # 方法1: 使用Xavier初始化
            if hasattr(torch.nn.init, 'xavier_uniform_'):
                new_weight = torch.empty_like(original_weight)
                torch.nn.init.xavier_uniform_(new_weight)
            else:
                # 方法2: 使用正态分布初始化
                new_weight = torch.randn_like(original_weight) * 0.1

            # 替换指定侧记忆池的权重
            adapter_layer.memory_weight[memory_pool_id] = new_weight.to(adapter_layer.device)

            # 同时重新初始化对应的激活统计
            if memory_pool_id < len(adapter_layer.memory_mean_act):
                adapter_layer.memory_mean_act[memory_pool_id] = EditingMeanAct()

        print(f"侧记忆池ID {memory_pool_id} 重新初始化完成")
        return True

    def process_merge_cnt_edits_with_evaluation(self, test_samples):
        """
        处理merge_cnt个数据编辑的完整流程：
        1. 记录编辑过程中的侧记忆池ID
        2. 进行推理评估
        3. 统计错误样本最多的侧记忆池ID
        4. 重新初始化该侧记忆池

        Args:
            test_samples: 测试样本列表，用于评估rewrite_acc

        Returns:
            dict: 包含评估结果和操作结果的字典
        """
        print("开始处理merge_cnt个数据编辑的完整流程...")

        # 步骤1: 记录编辑过程中的侧记忆池ID（已在edit方法中实现）
        print("步骤1: 侧记忆池ID记录功能已集成到编辑过程中")

        # 步骤2: 进行推理评估
        print("步骤2: 开始推理评估...")
        rewrite_acc, error_samples, error_memory_ids = self.evaluate_rewrite_accuracy(test_samples)

        # 步骤3: 统计错误样本最多的侧记忆池ID
        print("步骤3: 统计错误样本最多的侧记忆池ID...")
        most_error_memory_id, error_counts = self.find_most_error_prone_memory_pool(error_memory_ids)

        # 步骤4: 重新初始化该侧记忆池
        reinit_success = False
        if most_error_memory_id != -1:
            print("步骤4: 重新初始化错误最多的侧记忆池...")
            reinit_success = self.reinitialize_memory_pool(most_error_memory_id)
        else:
            print("步骤4: 跳过重新初始化（没有找到有效的侧记忆池ID）")

        # 返回结果
        result = {
            'rewrite_acc': rewrite_acc,
            'error_samples_count': len(error_samples),
            'error_samples': error_samples,  # 新增：返回实际的错误样本
            'error_memory_ids': error_memory_ids,  # 新增：返回错误样本对应的侧记忆池ID
            'most_error_memory_id': most_error_memory_id,
            'error_counts': error_counts,
            'reinit_success': reinit_success,
            'edit_memory_records': self.get_adapter_layer().edit_memory_records
        }

        print("完整流程处理完成！")
        print(f"结果摘要:")
        print(f"  - 重写准确率: {rewrite_acc:.4f}")
        print(f"  - 错误样本数量: {len(error_samples)}")
        print(f"  - 错误最多的侧记忆池ID: {most_error_memory_id}")
        print(f"  - 重新初始化成功: {reinit_success}")

        return result

    def collect_and_recombine_error_samples(self, error_samples, edit_data, merge_cnt):
        """
        收集错误样本并与edit_data重新组合成merge_cnt个样本

        Args:
            error_samples: 错误样本列表
            edit_data: 原始编辑数据
            merge_cnt: 每次编辑的样本数量

        Returns:
            new_edit_batch: 重新组合的编辑批次
            remaining_data: 剩余的数据
        """
        print(f"收集错误样本并重新组合，错误样本数量: {len(error_samples)}")

        # 从edit_data中移除已经编辑过的样本（这里简化处理，实际可能需要更复杂的逻辑）
        remaining_data = edit_data.copy()

        # 创建新的编辑批次
        new_edit_batch = []

        # 首先添加错误样本
        for error_sample in error_samples:
            if len(new_edit_batch) < merge_cnt:
                new_edit_batch.append(error_sample)

        # 如果错误样本不够merge_cnt个，从剩余数据中补充
        if len(new_edit_batch) < merge_cnt:
            needed_count = merge_cnt - len(new_edit_batch)
            additional_samples = remaining_data[:needed_count]
            new_edit_batch.extend(additional_samples)
            remaining_data = remaining_data[needed_count:]

        print(f"重新组合完成，新批次样本数量: {len(new_edit_batch)}")
        print(f"剩余数据数量: {len(remaining_data)}")

        return new_edit_batch, remaining_data

    def iterative_training_with_evaluation(self, edit_data, test_samples, merge_cnt, max_iterations=10,
                                           save_state_path=None):
        """
        迭代训练循环：编辑-评估-重新初始化-重新组合

        Args:
            edit_data: 原始编辑数据
            test_samples: 测试样本
            merge_cnt: 每次编辑的样本数量
            max_iterations: 最大迭代次数
            save_state_path: 状态保存路径

        Returns:
            dict: 包含所有迭代结果的字典
        """
        print(f"开始迭代训练，总数据量: {len(edit_data)}, 每次编辑: {merge_cnt}个样本")

        all_results = []
        remaining_data = edit_data.copy()
        iteration = 0

        while len(remaining_data) > 0 and iteration < max_iterations:
            iteration += 1
            print(f"\n=== 第 {iteration} 次迭代 ===")

            # 准备当前批次的编辑数据
            if len(remaining_data) >= merge_cnt:
                current_batch = remaining_data[:merge_cnt]
                remaining_data = remaining_data[merge_cnt:]
            else:
                current_batch = remaining_data
                remaining_data = []

            print(f"当前批次样本数量: {len(current_batch)}")

            # 执行编辑（这里需要调用实际的编辑方法）
            # 注意：这里需要根据实际的编辑接口进行调整
            print("执行编辑...")
            # editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

            # 执行评估和重新初始化
            print("执行评估和重新初始化...")
            result = self.process_merge_cnt_edits_with_evaluation(test_samples)

            # 记录本次迭代结果
            iteration_result = {
                'iteration': iteration,
                'batch_size': len(current_batch),
                'rewrite_acc': result['rewrite_acc'],
                'error_samples_count': result['error_samples_count'],
                'most_error_memory_id': result['most_error_memory_id'],
                'reinit_success': result['reinit_success'],
                'remaining_data_count': len(remaining_data)
            }
            all_results.append(iteration_result)

            print(f"第 {iteration} 次迭代完成:")
            print(f"  - 重写准确率: {result['rewrite_acc']:.4f}")
            print(f"  - 错误样本数量: {result['error_samples_count']}")
            print(f"  - 重新初始化成功: {result['reinit_success']}")
            print(f"  - 剩余数据: {len(remaining_data)}")

            # 保存状态（如果指定了保存路径）
            if save_state_path:
                temp_results = {
                    'total_iterations': iteration,
                    'final_rewrite_acc': result['rewrite_acc'],
                    'total_samples_processed': sum(r['batch_size'] for r in all_results),
                    'iteration_results': all_results
                }
                self.save_iterative_training_state(save_state_path, temp_results, iteration)

            # 如果重写准确率达到100%，可以提前结束
            if result['rewrite_acc'] >= 1.0:
                print("重写准确率达到100%，提前结束迭代")
                break

            # 如果还有剩余数据，收集错误样本并重新组合
            if len(remaining_data) > 0 and result['error_samples_count'] > 0:
                print("收集错误样本并重新组合...")
                # 从result中提取错误样本
                error_samples = result['error_samples']
                new_batch, remaining_data = self.collect_and_recombine_error_samples(
                    error_samples, remaining_data, merge_cnt
                )
                # 将新批次添加到剩余数据的前面
                remaining_data = new_batch + remaining_data

        print(f"\n迭代训练完成，总共进行了 {iteration} 次迭代")

        # 计算总体统计
        total_results = {
            'total_iterations': iteration,
            'final_rewrite_acc': all_results[-1]['rewrite_acc'] if all_results else 0,
            'total_samples_processed': sum(r['batch_size'] for r in all_results),
            'iteration_results': all_results
        }

        return total_results

    def save_iterative_training_state(self, save_path, iterative_results, current_iteration):
        """
        保存迭代训练的状态

        Args:
            save_path: 保存路径
            iterative_results: 迭代结果
            current_iteration: 当前迭代次数
        """
        import os
        import json

        state = {
            'current_iteration': current_iteration,
            'iterative_results': iterative_results,
            'edit_memory_records': self.get_adapter_layer().edit_memory_records,
            'current_edit_batch': self.get_adapter_layer().current_edit_batch,
            'memory_weight_count': len(self.get_adapter_layer().memory_weight),
            'merge_cnt': self.get_adapter_layer().merge_cnt
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4, ensure_ascii=False)

        print(f"迭代训练状态已保存到: {save_path}")

    def load_iterative_training_state(self, load_path):
        """
        加载迭代训练的状态

        Args:
            load_path: 加载路径

        Returns:
            dict: 加载的状态
        """
        import json

        if not os.path.exists(load_path):
            print(f"状态文件不存在: {load_path}")
            return None

        with open(load_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # 恢复状态
        self.get_adapter_layer().edit_memory_records = state.get('edit_memory_records', [])
        self.get_adapter_layer().current_edit_batch = state.get('current_edit_batch', [])

        print(f"迭代训练状态已从 {load_path} 加载")
        return state


class WISEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(WISEAdapter, self).__init__()
        self.selected_memory_ID = []  # add by wys
        # 新增：记录merge_cnt个编辑过程中的侧记忆池ID
        self.edit_memory_records = []  # 记录每次编辑使用的侧记忆池ID
        self.current_edit_batch = []  # 当前批次的编辑记录

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        print(f'config.retrieve模式为:{self.config.retrieve}')  # wys
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)
        self.memory_weight = []
        self.memory_mean_act = []
        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias  # For Conv1D
        else:
            self.bias = None
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

        # 评测期上下文与日志 cursor
        self.eval_tag = None
        self.eval_prompt = None
        self.activation_log_file = os.path.join('logs', 'activation_scores.txt')
        self.model_output_log_file = os.path.join('logs', 'model_outputs.txt')

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    # wys
    def add_elements(self, ele, ele1):
        self.memory_weight.insert(0, ele)
        self.memory_mean_act.insert(0, ele1)

    def wys(self):
        pass

    def save_weight(self):
        self.memory_weight.append(copy.deepcopy(self.new_weight))

        #
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute(
                        [self.config.weights / len(self.memory_weight) for _ in range(len(self.memory_weight))],
                        self.original_layer.weight, self.memory_weight, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute(
                        [0.4 / len(self.memory_weight) for _ in range(len(self.memory_weight))] + [0.6],
                        self.original_layer.weight, self.memory_weight + [self.layer.weight],
                        densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)],
                                                     self.original_layer.weight, self.memory_weight[-merge_num:],
                                                     densities=self.config.densities)
                ###wys

                ####
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    with open('edit_act.txt', 'a', encoding='utf-8') as f:
                        f.write(f'{edit_act.value_all}\n')
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def save_editing_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # 获取未被遮罩的元素索引
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # 更新遮罩状态
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def generate_fixed_mask(self, mask_line):  ##wys
        p_grad = self.new_weight.reshape(-1)
        mask = torch.zeros_like(p_grad, dtype=torch.float32)
        mask[:mask_line * 4096] = 1.0
        # %
        # for i in range(1,len(mask)//4096,2):
        #     mask[(i-1)*4096:(i)*4096] = 1.0

        # %
        p_mask = mask
        p_mask = p_mask.to(p_grad.device)
        self.weight_mask = p_mask

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight) if self.bias is None else torch.addmm(self.bias,
                                                                                      input.view(-1, input.size(-1)),
                                                                                      self.new_weight).view(
            input.size()[:-1] + (self.layer.nf,))

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def forward(self, *args):
        if self.editing:
            layer_out = self.new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            log_payload = None  #cursor
            if not self.config.retrieve:
                original_layer_output = self.original_layer(*args)
                layer_output = self.layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                dist1 = euc(original_layer_output, layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                if dist1.item() < threshold and dist2.item() < threshold:
                    layer_out = original_layer_output
                    chosen = 'original' #cursor
                elif dist1.item() > dist2.item():
                    layer_out = layer_output
                    chosen = 'layer' #cursor
                else:
                    layer_out = new_weight_layer_output
                    chosen = 'new' #wys
                    # 记录日志（非 retrieve）cursor
                    log_payload = {
                        'mode': 'inference',
                        'retrieve': False,
                        'eval_tag': getattr(self, 'eval_tag', None),
                        'prompt': getattr(self, 'eval_prompt', None),
                        'dist_to_layer': float(dist1.item()),
                        'dist_to_new': float(dist2.item()),
                        'threshold': float(threshold),
                        'chosen': chosen,
                }
            else:
                original_layer_output = self.original_layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist1 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio
                min_dist = dist1
                # print(f'threshold为：{threshold}')
                # print(f'当前min_dist为：{min_dist}')
                if min_dist.dim() > 0:
                    min_dist = min_dist.mean()
                if min_dist.item() < threshold:
                    layer_out = original_layer_output
                    chosen = 'original' #cursor

                else:
                    layer_out = new_weight_layer_output
                    chosen = 'new' #cursor

                temp = {}  # wys
                if len(self.memory_weight) < 1:
                    self.selected_memory_ID.append(-1)  # add by wys
                    memory_chosen = -1 #cursor
                for i in range(len(self.memory_weight)):
                    memory_retrieve_weight = self.memory_weight[i]
                    memory_weight_layer_output = F.linear(*args, memory_retrieve_weight)
                    dist = euc(original_layer_output, memory_weight_layer_output, self.config, infer=True)
                    # if dist >= min_dist and dist > self.memory_mean_act[i].min_act() * self.config.act_ratio:#wys
                    if dist > min_dist and dist > self.memory_mean_act[i].min_act() * self.config.act_ratio:
                        layer_out = memory_weight_layer_output
                        min_dist = dist
                        temp[str(dist)] = i  # wys

                try:
                    memory_id = temp[str(min_dist)]
                    self.selected_memory_ID.append(memory_id)  # add by wys
                    # 新增：记录到当前编辑批次中
                    self.current_edit_batch.append(memory_id)
                    chosen = f'memory_{memory_id}' #cursor
                    memory_chosen = memory_id
                except:
                    self.selected_memory_ID.append(-2)  # add by wys
                    # 新增：记录到当前编辑批次中
                    self.current_edit_batch.append(-2)
                    memory_chosen = -2 #cursor

                print(f'记录的ID为：{len(self.selected_memory_ID)}--{self.selected_memory_ID}')
                # add by wys

                data = {"num": len(self.selected_memory_ID), "all_layer": len(self.selected_memory_ID),
                        "selected_layer": [j for i, j in enumerate(self.selected_memory_ID) if i % 3 == 0]}
                print('当前路径：', os.getcwd())
                with open("selected_layer_recoder.json", "w") as f:
                    json.dump(data, f, indent=4)

                    # 记录日志（retrieve） #cursor
                log_payload = {
                    'mode': 'inference',
                    'retrieve': True,
                    'eval_tag': getattr(self, 'eval_tag', None),
                    'prompt': getattr(self, 'eval_prompt', None),
                    'dist_to_new': float(dist1.item()),
                    'threshold': float(threshold),
                    'min_dist_after_memory': float(min_dist.item()) if hasattr(min_dist, 'item') else float(
                        min_dist),
                    'chosen': chosen,
                    'memory_chosen': memory_chosen if 'memory_chosen' in locals() else None,
                }

        print('************************')
        # 将日志写入文件（若设置了 eval_tag，表示处于评测上下文）cursor
        try:
            if log_payload is not None and getattr(self, 'eval_tag', None) is not None:
                os.makedirs(os.path.dirname(self.activation_log_file), exist_ok=True)
                with open(self.activation_log_file, 'a', encoding='utf-8') as f:
                    json.dump(log_payload, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            # 避免影响推理主流程
            pass
        return layer_out


class WISEMultimodal(WISE):
    def edit(self, config, multimodal_inputs, text_tokens, ans_token_len, act_mask=None, deact_mask=None):
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}": v1.to('cpu') for k1, v1 in text_tokens.items()}, False])
        last_prompt_token_loc = (text_tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
            self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

            # --- train Wise value ---
        self.loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([super().get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len)

            act_loss = super()._cal_activation_loss(super().get_adapter_layer().original_layer_output,
                                                    super().get_adapter_layer().new_weight_layer_output,
                                                    config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)

            if self.loss_meter.stop():
                super().get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                super().get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_):  ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = super()._cal_memory_neg_activation_loss(
                        super().get_adapter_layer().original_layer_output,
                        super().get_adapter_layer().new_weight_layer_output, config=config,
                        act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}": v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = super()._cal_memory_pos_activation_loss(
                        super().get_adapter_layer().original_layer_output,
                        super().get_adapter_layer().new_weight_layer_output, config=config,
                        act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            super().get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and super().get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            self.loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                super()._norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            super().get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            super().get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def _cal_ft_loss(self, multimodal_inputs, text_tokens, last_prompt_token_loc, ans_token_len):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1

        if k != 1:
            raise AssertionError("Not support Batch Edit")

        bs = text_tokens["input_ids"].shape[0] - k
        logits = self.model(**multimodal_inputs).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = multimodal_inputs['input_ids'][:-k, 1:].contiguous()
        # only cal loss of target text tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        a = shift_logits.view(-1, shift_logits.size(-1))
        b = shift_labels.view(-1)[-ans_token_len:]
        a = a[-b.size(0):, :]
        loss = loss_fct(a, b)
        loss = loss.view(bs, -1)
        label_mask = torch.ones_like(loss, dtype=torch.bool)
        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss
