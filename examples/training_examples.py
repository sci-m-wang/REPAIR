# import os.path
# import sys
# import json
# import argparse
# import pickle
# sys.path.append('..')
# from easyeditor import (
#     FTHyperParams,
#     GraceHyperParams,
#     MEMITHyperParams,
#     ROMEHyperParams,
#     MENDHyperParams,
#     WISEHyperParams,
#     BaseEditor,
#     summary_metrics,
# )
#
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
# def run_iterative_training_example():
#     """
#     运行迭代训练示例
#     """
#     print("=== WISE迭代训练示例 ===")
#
#     # 解析命令行参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--hparams_dir', default='../hparams/WISE/llama-3-8b.yaml', type=str)
#     parser.add_argument('--data_dir', default='../data/wise', type=str)
#     parser.add_argument('--data_type', default='ZsRE', type=str, choices=['ZsRE', 'temporal', 'hallucination'])
#     parser.add_argument('--output_dir', default='./outputs', type=str)
#     parser.add_argument('--ds_size', default=20, type=int, help="总数据量")
#     parser.add_argument('--merge_cnt', default=5, type=int, help="每次编辑的样本数量")
#     parser.add_argument('--max_iterations', default=5, type=int, help="最大迭代次数")
#     parser.add_argument('--save_state_path', default='./outputs/iterative_state.json', type=str, help="状态保存路径")
#
#     args = parser.parse_args()
#
#     # 加载数据
#     print(f"加载 {args.data_type} 数据...")
#     if args.data_type == 'ZsRE':
#         edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
#         loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
#         loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]
#
#         prompts = [edit_data_['src'] for edit_data_ in edit_data]
#         subject = [edit_data_['subject'] for edit_data_ in edit_data]
#         rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
#         target_new = [edit_data_['alt'] for edit_data_ in edit_data]
#         locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
#         locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
#         locality_inputs = {
#             'neighborhood': {
#                 'prompt': locality_prompts,
#                 'ground_truth': locality_ans
#             },
#         }
#     elif args.data_type == 'hallucination':
#         edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[
#                     :K]
#         loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[
#                    :K]
#         loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in
#                        loc_data]
#
#         prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
#         subject = [edit_data_['subject'] for edit_data_ in edit_data]
#         rephrase_prompts = None
#         target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
#         locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
#         locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
#         locality_inputs = {
#             'neighborhood': {
#                 'prompt': locality_prompts,
#                 'ground_truth': locality_ans
#             },
#         }
#     elif args.data_type == 'temporal':
#         edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-edit.json', 'r', encoding='utf-8'))[:K]
#         loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-train.json', 'r', encoding='utf-8'))[:K]
#         loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in
#                        loc_data]
#
#         prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
#         subject = [edit_data_['subject'] for edit_data_ in edit_data]
#         rephrase_prompts = [edit_data_['ood_rephrase'] for edit_data_ in edit_data]
#         target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
#         locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
#         locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
#         locality_inputs = {
#             'neighborhood': {
#                 'prompt': locality_prompts,
#                 'ground_truth': locality_ans
#             },
#         }
#     print(f"加载了 {len(edit_data)} 个编辑样本")
#
#     # 加载配置
#     hparams = WISEHyperParams.from_hparams(args.hparams_dir)
#
#     # 创建编辑器
#     editor = BaseEditor.from_hparams(hparams)
#
#     # 准备测试样本
#     print("准备测试样本...")
#     test_samples = []
#     for i, data in enumerate(edit_data):
#         if args.data_type == 'ZsRE':
#             prompt = data['src']
#             target = data['alt']
#         elif args.data_type == 'hallucination':
#             prompt = data['prompt']
#             target = data['target_new']
#         elif args.data_type == 'temporal':
#             prompt = data['prompt']
#             target = data['target_new']
#
#         # test_sample = {
#         #     'input_ids': editor.editor.model.tokenizer.encode(prompt, return_tensors='pt'),
#         #     'expected_output': editor.editor.model.tokenizer.encode(target, add_special_tokens=False)
#         # }
#         test_sample = {
#             'input_ids': editor.tok.encode(prompt, return_tensors='pt'),
#             'expected_output': editor.tok.encode(target, add_special_tokens=False)
#         }
#         test_samples.append(test_sample)
#
#     #test_samples=edit_data
#
#     print(f"准备了 {len(test_samples)} 个测试样本")
#
#     # 准备编辑请求
#     print("准备编辑请求...")
#     edit_requests = []
#     for data in edit_data:
#         if args.data_type == 'ZsRE':
#             request = {
#                 'prompt': data['src'],
#                 'target_new': data['alt'],
#                 'subject': data['subject']
#             }
#         elif args.data_type == 'hallucination':
#             request = {
#                 'prompt': data['prompt'],
#                 'target_new': data['target_new'],
#                 'subject': data['subject']
#             }
#         elif args.data_type == 'temporal':
#             request = {
#                 'prompt': data['prompt'],
#                 'target_new': data['target_new'],
#                 'subject': data['subject']
#             }
#         edit_requests.append(request)
#
#     print(f"准备了 {len(edit_requests)} 个编辑请求")
#
#     # 执行迭代训练
#     print(f"\n开始迭代训练...")
#     print(f"总数据量: {len(edit_requests)}")
#     print(f"每次编辑: {args.merge_cnt} 个样本")
#     print(f"最大迭代次数: {args.max_iterations}")
#     #
#     # iterative_results = editor.editor.iterative_training_with_evaluation(
#     #     edit_data=edit_requests,
#     #     test_samples=test_samples,
#     #     merge_cnt=args.merge_cnt,
#     #     max_iterations=args.max_iterations,
#     #     save_state_path=args.save_state_path
#     # )
#     # 手动实现迭代训练循环
#     all_results = []
#     remaining_data = edit_requests.copy()
#     iteration = 0
#
#     while len(remaining_data) > 0 and iteration < args.max_iterations:
#         iteration += 1
#         print(f"\n=== 第 {iteration} 次迭代 ===")
#
#         # 准备当前批次的编辑数据
#         if len(remaining_data) >= args.merge_cnt:
#             current_batch = remaining_data[:args.merge_cnt]
#             remaining_data = remaining_data[args.merge_cnt:]
#         else:
#             current_batch = remaining_data
#             remaining_data = []
#
#         print(f"当前批次样本数量: {len(current_batch)}")
#
#         # 执行编辑
#         print("执行编辑...")
#         for i, request in enumerate(current_batch):
#             print(f"编辑样本 {i + 1}/{len(current_batch)}: {request['prompt'][:50]}...")
#             editor.edit(
#                 prompts=request['prompt'],
#                 target_new=request['target_new'],
#                 ground_truth=request.get('ground_truth', '<|endoftext|>'),
#                 verbose=False
#             )
#
#         # 执行评估和重新初始化
#         print("执行评估和重新初始化...")
#         result = editor.apply_algo.process_merge_cnt_edits_with_evaluation(test_samples)
#
#         # 记录本次迭代结果
#         iteration_result = {
#             'iteration': iteration,
#             'batch_size': len(current_batch),
#             'rewrite_acc': result['rewrite_acc'],
#             'error_samples_count': result['error_samples_count'],
#             'most_error_memory_id': result['most_error_memory_id'],
#             'reinit_success': result['reinit_success'],
#             'remaining_data_count': len(remaining_data)
#         }
#         all_results.append(iteration_result)
#
#         print(f"第 {iteration} 次迭代完成:")
#         print(f"  - 重写准确率: {result['rewrite_acc']:.4f}")
#         print(f"  - 错误样本数量: {result['error_samples_count']}")
#         print(f"  - 重新初始化成功: {result['reinit_success']}")
#         print(f"  - 剩余数据: {len(remaining_data)}")
#
#         # 如果重写准确率达到100%，可以提前结束
#         if result['rewrite_acc'] >= 1.0:
#             print("重写准确率达到100%，提前结束迭代")
#             break
#
#         # 如果还有剩余数据，收集错误样本并重新组合
#         if len(remaining_data) > 0 and result['error_samples_count'] > 0:
#             print("收集错误样本并重新组合...")
#             error_samples = result['error_samples']
#             new_batch, remaining_data = editor.apply_algo.collect_and_recombine_error_samples(
#                 error_samples, remaining_data, args.merge_cnt
#             )
#             # 将新批次添加到剩余数据的前面
#             remaining_data = new_batch + remaining_data
#
#     print(f"\n迭代训练完成，总共进行了 {iteration} 次迭代")
#
#     # 计算总体统计
#     iterative_results = {
#         'total_iterations': iteration,
#         'final_rewrite_acc': all_results[-1]['rewrite_acc'] if all_results else 0,
#         'total_samples_processed': sum(r['batch_size'] for r in all_results),
#         'iteration_results': all_results
#     }
#     # 保存结果
#     os.makedirs(args.output_dir, exist_ok=True)
#     output_file = os.path.join(args.output_dir, f'iterative_training_results_{args.data_type}.json')
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(iterative_results, f, indent=4, ensure_ascii=False)
#
#     # 打印结果摘要
#     print(f"\n=== 迭代训练完成 ===")
#     print(f"总迭代次数: {iterative_results['total_iterations']}")
#     print(f"最终重写准确率: {iterative_results['final_rewrite_acc']:.4f}")
#     print(f"总处理样本数: {iterative_results['total_samples_processed']}")
#     print(f"结果已保存到: {output_file}")
#
#     # 打印每次迭代的详细结果
#     print(f"\n=== 每次迭代结果 ===")
#     for i, result in enumerate(iterative_results['iteration_results']):
#         print(f"第 {result['iteration']} 次迭代:")
#         print(f"  - 批次大小: {result['batch_size']}")
#         print(f"  - 重写准确率: {result['rewrite_acc']:.4f}")
#         print(f"  - 错误样本数量: {result['error_samples_count']}")
#         print(f"  - 重新初始化成功: {result['reinit_success']}")
#         print(f"  - 剩余数据: {result['remaining_data_count']}")
#         print()
#
#
# if __name__ == "__main__":
#     run_iterative_training_example()
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from transformers import LlamaTokenizer,PreTrainedTokenizerFast, LlamaTokenizerFast
import os.path
import sys
import json
import argparse
import pickle
sys.path.append('/data/REPAIR')
from easyeditor.models.wise.WISE_V1 import WISE
from easyeditor import (
    FTHyperParams,
    GraceHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    MENDHyperParams,
    WISEHyperParams,
    BaseEditor,
    summary_metrics,
)
# 安全获取 WISE 实例（兼容 function/bound-method 两种情况）
def get_wise(editor):
    # 情况A：apply_algo 是绑定方法（少数环境），可从 __self__ 拿到实例
    algo = getattr(editor, 'apply_algo', None)
    if hasattr(algo, '__self__') and isinstance(algo.__self__, WISE):
        return algo.__self__
    # 情况B：无法从 apply_algo 拿到实例，则显式构造一个 WISE 实例
    return WISE(editor.hparams, editor.model, editor.hparams.device)


def iterative_training_with_evaluation(editor, edit_data, test_samples, merge_cnt, max_iterations=10,
                                       save_state_path=None):
    """
    迭代训练循环：编辑-评估-重新初始化-重新组合

    Args:
        editor: BaseEditor实例，用于执行编辑操作
        edit_data: 原始编辑数据列表，每个元素包含 'prompt', 'target_new', 'ground_truth' 等字段
        test_samples: 测试样本列表，用于评估rewrite_acc，每个元素包含 'input_ids', 'expected_output'
        merge_cnt: 每次编辑的样本数量
        max_iterations: 最大迭代次数
        save_state_path: 状态保存路径（可选）

    Returns:
        dict: 包含所有迭代结果的字典，包含以下字段：
            - total_iterations: 总迭代次数
            - final_rewrite_acc: 最终重写准确率
            - total_samples_processed: 总处理样本数
            - iteration_results: 每次迭代的详细结果列表
    """
    import os
    import json

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

        # 执行编辑
        print("执行编辑...")
        for i, request in enumerate(current_batch):
            print(f"编辑样本 {i + 1}/{len(current_batch)}: {request['prompts'][:50]}...")
            editor.edit(
                # prompts=request['prompt'],
                # target_new=request['target_new'],
                # ground_truth=request.get('ground_truth', '<|endoftext|>'),
                # verbose=False
                prompts=request['prompts'],
                rephrase_prompts=request['rephrase_prompts'],
                target_new=request['target_new'],
                loc_prompts=request['loc_prompts'],
                subject=request['subject'],
                locality_inputs=request['locality_inputs'],
                sequential_edit=request['sequential_edit'],
                eval_metric=request['data_type']
            )

        # 执行评估和重新初始化
        print("执行评估和重新初始化...")
        #result = editor.apply_algo.process_merge_cnt_edits_with_evaluation(test_samples)
        wise = get_wise(editor)
        result = wise.process_merge_cnt_edits_with_evaluation(test_samples)

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
            #editor.apply_algo.save_iterative_training_state(save_state_path, temp_results, iteration)
            wise.save_iterative_training_state(save_state_path, temp_results, iteration)

        # 如果重写准确率达到100%，可以提前结束
        if result['rewrite_acc'] >= 1.0:
            print("重写准确率达到100%，提前结束迭代")
            break

        # 如果还有剩余数据，收集错误样本并重新组合
        if len(remaining_data) > 0 and result['error_samples_count'] > 0:
            print("收集错误样本并重新组合...")
            error_samples = result['error_samples']
            new_batch, remaining_data = wise.collect_and_recombine_error_samples(
                error_samples, remaining_data, merge_cnt
            )
            # 将新批次添加到剩余数据的前面
            remaining_data = new_batch + remaining_data

    print(f"\n迭代训练完成，总共进行了 {iteration} 次迭代")

    # 计算总体统计
    iterative_results = {
        'total_iterations': iteration,
        'final_rewrite_acc': all_results[-1]['rewrite_acc'] if all_results else 0,
        'total_samples_processed': sum(r['batch_size'] for r in all_results),
        'iteration_results': all_results
    }

    return iterative_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=False, type=str,default='WISE')
    parser.add_argument('--hparams_dir', required=False, type=str,default='/data/REPAIR/hparams/WISE/llama-3-8b.yaml')  #default='../hparams/WISE/llama-3-8b.yaml')
    parser.add_argument('--data_dir', required=False, type=str,default='/data/REPAIR/data/wise')
    parser.add_argument('--data_type', required=False, type=str,
                        choices=['ZsRE', 'temporal', 'hallucination'],default='ZsRE')
    parser.add_argument('--output_dir', default='/data/REPAIR/examples/outputs', type=str)
    parser.add_argument('--ds_size', default=120, type=int)
    parser.add_argument('--sequential_edit', action="store_true",default=True)


    args = parser.parse_args()

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    else:
        raise NotImplementedError

    K =args.ds_size


    if args.data_type == 'ZsRE':
        #待编辑数据，后续可以用来转
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
        #评估编辑后模型是否影响到原有知识的数据--局部性评估
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
        #将事实拼接起来，为了更方便进行局部性评估
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

        prompts = [edit_data_['src'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        #用于测试模型在学会新事实后，能否将新事实用不同的表达方式说出来
        rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
        #模型需要学习到的新的事实
        target_new = [edit_data_['alt'] for edit_data_ in edit_data]
        #局部性评估的输入
        locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood':{
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'hallucination':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/hallucination-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = None
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }
    elif args.data_type == 'temporal':
        edit_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-edit.json', 'r', encoding='utf-8'))[:K]
        loc_data = json.load(open(f'{args.data_dir}/{args.data_type}/temporal-train.json', 'r', encoding='utf-8'))[:K]
        loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in loc_data]

        prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
        subject = [edit_data_['subject'] for edit_data_ in edit_data]
        rephrase_prompts = [edit_data_['ood_rephrase'] for edit_data_ in edit_data]
        target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
        locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
        locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
        locality_inputs = {
            'neighborhood': {
                'prompt': locality_prompts,
                'ground_truth': locality_ans
            },
        }

    #%test_samples
    test_samples_data=json.load(open(f'{args.data_dir}/{args.data_type}/zsre_mend_edit.json', 'r', encoding='utf-8'))[:20]
    test_samples_prompts = [edit_data_['src'] for edit_data_ in test_samples_data]
    test_samples_subject = [edit_data_['subject'] for edit_data_ in test_samples_data]
    test_samples_rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in test_samples_data]
    test_samples_target_new = [edit_data_['alt'] for edit_data_ in test_samples_data]
    test_samples_locality_prompts = [edit_data_['loc'] for edit_data_ in test_samples_data]
    test_samples_locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_samples_data]
    test_samples_locality_inputs = {
        'neighborhood': {
            'prompt': test_samples_locality_prompts,
            'ground_truth': test_samples_locality_ans
        },
    }


    hparams = editing_hparams.from_hparams(f'{args.hparams_dir}')

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
        )

    print("See results at: ", output_file)

    eval_metric = {
        'ZsRE': 'token em',     #准确性、词元精确匹配
        'hallucination': 'ppl', #困惑度
        'temporal': 'ood_ppl'   #泛化能力
    }

    #在这里加载基础编辑器的实例
    editor = BaseEditor.from_hparams(hparams)
    # metrics, edited_model, _ = editor.edit(
    #     prompts=prompts,
    #     rephrase_prompts=rephrase_prompts,
    #     target_new=target_new,
    #     loc_prompts=loc_prompts,
    #     subject=subject,
    #     locality_inputs=locality_inputs,
    #     sequential_edit=args.sequential_edit,
    #     eval_metric=eval_metric[args.data_type]
    # )


    # prompts = request['prompts'],
    # rephrase_prompts = request['rephrase_prompts'],
    # target_new = request['target_new'],
    # loc_prompts = request['loc_prompts'],
    # subject = request['subject'],
    # locality_inputs = request['locality_inputs'],
    # sequential_edit = request['sequential_edit'],
    # eval_metric = request['data_type']
    test_samples = []
    tok = AutoTokenizer.from_pretrained('/data/REPAIR/Meta-Llama-3-8B-Instruct')
    for i, data in enumerate(edit_data):
        if args.data_type == 'ZsRE':
            prompt = data['src']
            target = data['alt']
        elif args.data_type == 'hallucination':
            prompt = data['prompt']
            target = data['target_new']
        elif args.data_type == 'temporal':
            prompt = data['prompt']
            target = data['target_new']

        test_sample = {
            'input_ids': tok(prompt, return_tensors='pt'),
            'expected_output': tok(target, add_special_tokens=False)
        }
        test_samples.append(test_sample)

        #test_samples=edit_data

    print(f"准备了 {len(test_samples)} 个测试样本")
    edit_requests = []
    for data in edit_data:
        if args.data_type == 'ZsRE':
            request = {
                'prompts': data['src'],
                'rephrase_prompts': data['rephrase'],
                'target_new': data['alt'],
                'loc_prompts':data['loc'] + ' ' + data['loc_ans'],
                'subject': data['subject'],
                'locality_inputs':{'neighborhood':{'prompt':data['loc'],'ground_truth':
                                                   data['loc_ans']}},
                'sequential_edit':True,
                'data_type':'token em'
            }
        edit_requests.append(request)

    # test_samples = []
    # for data in test_samples_data:
    #     if args.data_type == 'ZsRE':
    #         request = {
    #             'prompts': data['src'],
    #             'rephrase_prompts': data['rephrase'],
    #             'target_new': data['alt'],
    #             'loc_prompts':data['loc'] + ' ' + data['loc_ans'],
    #             'subject': data['subject'],
    #             'locality_inputs':{'neighborhood':{'prompt':data['loc'],'ground_truth':
    #                                                data['loc_ans']}},
    #             'sequential_edit':True,
    #             'data_type':'token em'
    #         }
    #     test_samples.append(request)

    metrics_=iterative_training_with_evaluation(editor, edit_requests, test_samples, 30, max_iterations=10,
                                           save_state_path=None)

    with open(output_file, 'w') as f:
        json.dump(metrics_, f, indent=4)

    if len(metrics_) > 0:
        summary_metrics(metrics_)

