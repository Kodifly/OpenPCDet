import pickle
import time
import plotly.graph_objects as go
import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import matplotlib.pyplot as plt
import pandas as pd
import os


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])






def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)
    
    create_visuals_and_tables(result_dict['eval_metrics'],final_output_dir)


    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict



def plot_table_data(table_data, result_directory, plot_name):
    # Create a DataFrame from the table data
    df = pd.DataFrame(table_data)

    # Separate the DataFrame into two, based on the scale of the metrics
    df_count = df[['TP', 'FP', 'FN', 'num_gts']]
    df_performance = df[['precision', 'recall', 'F1-score']]

    # Drop duplicate rows
    df_count = df_count.drop_duplicates()
    df_performance = df_performance.drop_duplicates()

    # Plot the count metrics
    fig = go.Figure(data=[
        go.Bar(name=f'TP {df_count["TP"].mean():.3f}', x=df_count.index, y=df_count['TP']),
        go.Bar(name=f'FP {df_count["FP"].mean():.3f}', x=df_count.index, y=df_count['FP']),
        go.Bar(name=f'FN {df_count["FN"].mean():.3f}', x=df_count.index, y=df_count['FN']),
        go.Bar(name=f'num_gts {df_count["num_gts"].mean():.3f}', x=df_count.index, y=df_count['num_gts'])
    ])
    fig.update_layout(barmode='group', title_text=f'{plot_name} (Count Metrics)', xaxis_title="Difficulty", yaxis_title="Value")
    fig.write_image(f"{result_directory}/{plot_name}_count_metrics.png")

    # Plot the performance metrics
    fig = go.Figure(data=[
        go.Bar(name=f'precision {df_performance["precision"].mean():.3f}', x=df_performance.index, y=df_performance['precision']),
        go.Bar(name=f'recall {df_performance["recall"].mean():.3f}', x=df_performance.index, y=df_performance['recall']),
        go.Bar(name=f'F1-score {df_performance["F1-score"].mean():.3f}', x=df_performance.index, y=df_performance['F1-score'])
    ])
    fig.update_layout(barmode='group', title_text=f'{plot_name} (Performance Metrics)', xaxis_title="Difficulty", yaxis_title="Value")
    fig.write_image(f"{result_directory}/{plot_name}_performance_metrics.png")
def create_visuals_and_tables(data, result_directory):
    # First Level: class_names
    os.makedirs(result_directory,exist_ok=True)
    for class_name in data.keys():
        class_data = data[class_name]
        
        # Second Level: eval_metrics
        for eval_metric in class_data.keys():
            eval_data = class_data[eval_metric]
            
            # Third Level: IoU_threshold
            for iou_threshold in eval_data.keys():
                metrics_data = eval_data[iou_threshold]
                
                # Precision Recall Curve
                fig, ax = plt.subplots()
                ax.plot(metrics_data['recall'][0], metrics_data['precision'][0], label="0")
                ax.plot(metrics_data['recall'][1], metrics_data['precision'][1], label="1")
                ax.plot(metrics_data['recall'][2], metrics_data['precision'][2], label="2")
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'Precision Recall Curve for Class: {class_name}, Eval_metric: {eval_metric}, IoU: {iou_threshold}')
                ax.legend()
                fig.savefig(f"{result_directory}/{class_name}_{eval_metric}_{iou_threshold}_precision_recall_curve.png")
                
                # Metrics Table
                tp = metrics_data['TP']
                fp = metrics_data['FP']
                fn = metrics_data['FN']
                num_gts = metrics_data['num_gts']
                precision = {k: tp[k] / (tp[k] + fp[k]) for k in tp.keys()}
                recall = {k: tp[k] / (tp[k] + fn[k]) for k in tp.keys()}
                f1_score = {k: 2 * (precision[k] * recall[k]) / (precision[k] + recall[k]) for k in tp.keys()}
                
                table_data = {
                'TP': {k: int(v) for k, v in tp.items()},
                'FP': {k: int(v) for k, v in fp.items()},
                'FN': {k: int(v) for k, v in fn.items()},
                'num_gts': {k: int(v) for k, v in num_gts.items()},
                'precision': {k: round(v, 3) for k, v in precision.items()},
                'recall': {k: round(v, 3) for k, v in recall.items()},
                'F1-score': {k: round(v, 3) for k, v in f1_score.items()},
                }
                df = pd.DataFrame(table_data)
                plot_table_data(df, result_directory, f"{class_name}_{eval_metric}_{iou_threshold}_plot")
                df.to_csv(f"{result_directory}/{class_name}_{eval_metric}_{iou_threshold}_metrics_table.csv", index=False)



if __name__ == '__main__':
    pass
