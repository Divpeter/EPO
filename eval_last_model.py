import os
# from sklearn.metrics import log_loss, roc_auc_score
import random
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from librerank.utils import *
from librerank.reranker import *
from librerank.CMR_generator import *
from librerank.rl_reranker import *
from librerank.LAST_generator import *
from librerank.LAST_evaluator import *
from librerank.EPO import *
import datetime
from run_reranker import eval
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import json


# import seaborn as sns
def eval_last(model, data, batch_size, isrank, metric_scope, _print=False, evaluator=None):
    step_sizes = len(model.step_sizes)
    preds_cmr, preds_last = [], []
    losses = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    # eva_cmr_ave, eva_last_ave = [[] for _ in range(len(metric_scope))], [[] for _ in range(len(metric_scope))]
    eva_cmr_ave, eva_last_ave = [], []
    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        inference_order, inference_predict, loss, cate_seq, cate_chosen = model.inference(data_batch)
        losses.append(loss)
        rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, inference_order)
                            
        #labels = np.array(model.build_label_reward(data_batch[4], inference_order))
        #auc_rewards = np.array(model.build_ndcg_reward(labels))
        #base_auc_rewards = np.array(model.build_ndcg_reward(data_batch[4]))
        #auc_rewards -= base_auc_rewards
        auc_rewards = evaluator.predict_evaluator(np.array(data_batch[1]), rl_sp_outputs, rl_de_outputs, data_batch[6], data_batch)
        base_auc_rewards = evaluator.predict_evaluator(np.array(data_batch[1]), np.array(data_batch[2]), np.array(data_batch[3]),
                                             data_batch[6], data_batch)
        auc_rewards -= base_auc_rewards

        # _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)  # rank base rerank new
        # _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
        base_div_rewards = model.build_alpha_ndcg_reward(cate_seq, cate_seq)  # rank base rerank new
        div_rewards = model.build_alpha_ndcg_reward(cate_chosen, cate_seq)
        div_rewards -= base_div_rewards

        pred, order = model.instant_learning(data_batch, auc_rewards, div_rewards, inference_order)  # pred: [L, B, N]
        batch_ave_steps = []  # [L, scope_num, B]
        ave_collect = []
        for i in range(step_sizes):
            batch_sum, batch_ave = evaluator_metrics_ns(data_batch, order[i], metric_scope, evaluator)  # [scope_num, B]
            res = list(evaluate_multi(data_batch[4], pred[i], list(map(lambda a: [i[1] for i in a], data_batch[2])), metric_scope, isrank, _print))  # [3+2, scope_num, ]
            div_metrics = res[-1][-1]
            ndcg_metrics = res[1][-1]
            ave_collect.append(batch_ave)
            batch_ave = np.array(batch_ave)*0+np.array(ndcg_metrics)*1
            batch_ave_steps.append(batch_ave)
            #batch_ave_steps.append(ndcg_metrics)
        batch_ave_steps_cmr = np.array(batch_ave_steps[:step_sizes // 2])
        batch_ave_steps_last = np.array(batch_ave_steps[step_sizes // 2:])
        best_cmr_idx = np.argmax(batch_ave_steps_cmr[:, -1, :], axis=0)
        best_last_idx = np.argmax(batch_ave_steps_last[:, -1, :], axis=0)
        pred = np.array(pred) 
        pred_cmr = pred[best_cmr_idx, np.arange(pred.shape[1]), :]
        pred_last = pred[step_sizes//2 + best_last_idx, np.arange(pred.shape[1]), :]
        preds_cmr.extend(pred_cmr)
        preds_last.extend(pred_last)
        batch_cmr_metrics = batch_ave_steps_cmr[best_cmr_idx, :, np.arange(pred.shape[1])]
        batch_last_metrics = batch_ave_steps_last[best_last_idx, :, np.arange(pred.shape[1])]
        # for i in range(len(metric_scope)):
        # eva_cmr_ave.extend(batch_cmr_metrics)
        # eva_last_ave.extend(batch_last_metrics)

        batch_ave_steps_cmr_2 = np.array(ave_collect[:step_sizes // 2])
        batch_ave_steps_last_2 = np.array(ave_collect[step_sizes // 2:])
        best_cmr_idx_2 = np.argmax(batch_ave_steps_cmr_2[:, -1, :], axis=0)
        best_last_idx_2 = np.argmax(batch_ave_steps_last_2[:, -1, :], axis=0)
        pred_2 = np.array(pred) 
        # pred_cmr_2 = pred_2[best_cmr_idx_2, np.arange(pred_2.shape[1]), :]
        # pred_last_2 = pred_2[step_sizes//2 + best_last_idx_2, np.arange(pred_2.shape[1]), :]
        # preds_cmr.extend(pred_cmr)
        # preds_last.extend(pred_last)
        batch_cmr_metrics_2 = batch_ave_steps_cmr_2[best_cmr_idx_2, :, np.arange(pred_2.shape[1])]
        batch_last_metrics_2 = batch_ave_steps_last_2[best_last_idx_2, :, np.arange(pred_2.shape[1])]
        # for i in range(len(metric_scope)):
        eva_cmr_ave.extend(batch_cmr_metrics_2)
        eva_last_ave.extend(batch_last_metrics_2)

    labels = data[4]
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))

    print("Base model")
    res = list(evaluate_multi(labels, preds_cmr, cate_ids, metric_scope, isrank, _print))  # [3+2, scope_num, ]
    for j, s in enumerate(params.metric_scope):
        print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f  EVA_AVE: %.4f" % (
            s, res[0][j], res[1][j], res[2][j], res[4][j], res[5][j], np.mean(np.array(eva_cmr_ave), axis=0)[j]))
    # s, res[0][j], res[1][j], res[2][j], res[4][j], 10 * (np.mean(np.array(eva_cmr_ave), axis=0)[j] - 0.9 * res[4][j])))

    print("LAST")
    res = list(evaluate_multi(labels, preds_last, cate_ids, metric_scope, isrank, _print))  # [3+2, scope_num, ]
    for j, s in enumerate(params.metric_scope):
        print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f  EVA_AVE: %.4f" % (
            s, res[0][j], res[1][j], res[2][j], res[4][j], res[5][j], np.mean(np.array(eva_last_ave), axis=0)[j]))
    # s, res[0][j], res[1][j], res[2][j], res[4][j], 10 * (np.mean(np.array(eva_last_ave), axis=0)[j] - 0.9 * res[4][j])))

    print("EVAL TIME: %.4fs" % (time.time() - t))

def eval_epo(model, data, batch_size, isrank, metric_scope, _print=False, evaluator=None):
    step_sizes = len(model.step_sizes)
    preds_cmr, preds_last = [], []
    losses = []

    data_size = len(data[0])
    batch_num = data_size // batch_size
    print('eval', batch_size, batch_num)

    # eva_cmr_ave, eva_last_ave = [[] for _ in range(len(metric_scope))], [[] for _ in range(len(metric_scope))]
    eva_cmr_ave, eva_last_ave = [], []
    t = time.time()
    for batch_no in range(batch_num):
        data_batch = get_aggregated_batch(data, batch_size=batch_size, batch_no=batch_no)
        inference_order, inference_predict, loss, cate_seq, cate_chosen = model.inference(data_batch)
        losses.append(loss)
        rl_sp_outputs, rl_de_outputs = model.build_ft_chosen(data_batch, inference_order)
                            
        #labels = np.array(model.build_label_reward(data_batch[4], inference_order))
        #auc_rewards = np.array(model.build_ndcg_reward(labels))
        #base_auc_rewards = np.array(model.build_ndcg_reward(data_batch[4]))
        #auc_rewards -= base_auc_rewards
        auc_rewards = evaluator.predict_evaluator(np.array(data_batch[1]), rl_sp_outputs, rl_de_outputs, data_batch[6], data_batch)
        base_auc_rewards = evaluator.predict_evaluator(np.array(data_batch[1]), np.array(data_batch[2]), np.array(data_batch[3]),
                                             data_batch[6], data_batch)
        auc_rewards -= base_auc_rewards

        # _, base_div_rewards = model.build_erria_reward(cate_seq, cate_seq)  # rank base rerank new
        # _, div_rewards = model.build_erria_reward(cate_chosen, cate_seq)
        base_div_rewards = model.build_alpha_ndcg_reward(cate_seq, cate_seq)  # rank base rerank new
        div_rewards = model.build_alpha_ndcg_reward(cate_chosen, cate_seq)
        div_rewards -= base_div_rewards

        pred, order = model.instant_learning(data_batch, auc_rewards, div_rewards, inference_order)  # pred: [L, B, N]
        batch_ave_steps = []  # [L, scope_num, B]
        ave_collect = []
        for i in range(step_sizes):
            batch_sum, batch_ave = evaluator_metrics_ns(data_batch, order[i], metric_scope, evaluator)  # [scope_num, B]
            res = list(evaluate_multi(data_batch[4], pred[i], list(map(lambda a: [i[1] for i in a], data_batch[2])), metric_scope, isrank, _print))  # [3+2, scope_num, ]
            div_metrics = res[-1][-1]
            ndcg_metrics = res[1][-1]
            ave_collect.append(batch_ave)
            #decide whether the model use evaluator score or ndcg as reward
            batch_ave = np.array(batch_ave)*0+np.array(ndcg_metrics)*1
            batch_ave_steps.append(batch_ave)
            #batch_ave_steps.append(ndcg_metrics)
        batch_ave_steps_cmr = np.array(batch_ave_steps[:step_sizes // 2])
        batch_ave_steps_last = np.array(batch_ave_steps[step_sizes // 2:])
        best_cmr_idx = np.argmax(batch_ave_steps_cmr[:, -1, :], axis=0)
        best_last_idx = np.argmax(batch_ave_steps_last[:, -1, :], axis=0)
        pred = np.array(pred) 
        pred_cmr = pred[best_cmr_idx, np.arange(pred.shape[1]), :]
        pred_last = pred[step_sizes//2 + best_last_idx, np.arange(pred.shape[1]), :]
        preds_cmr.extend(pred_cmr)
        preds_last.extend(pred_last)
        batch_cmr_metrics = batch_ave_steps_cmr[best_cmr_idx, :, np.arange(pred.shape[1])]
        batch_last_metrics = batch_ave_steps_last[best_last_idx, :, np.arange(pred.shape[1])]
        # for i in range(len(metric_scope)):
        # eva_cmr_ave.extend(batch_cmr_metrics)
        # eva_last_ave.extend(batch_last_metrics)

        batch_ave_steps_cmr_2 = np.array(ave_collect[:step_sizes // 2])
        batch_ave_steps_last_2 = np.array(ave_collect[step_sizes // 2:])
        best_cmr_idx_2 = np.argmax(batch_ave_steps_cmr_2[:, -1, :], axis=0)
        best_last_idx_2 = np.argmax(batch_ave_steps_last_2[:, -1, :], axis=0)
        pred_2 = np.array(pred) 
        # pred_cmr_2 = pred_2[best_cmr_idx_2, np.arange(pred_2.shape[1]), :]
        # pred_last_2 = pred_2[step_sizes//2 + best_last_idx_2, np.arange(pred_2.shape[1]), :]
        # preds_cmr.extend(pred_cmr)
        # preds_last.extend(pred_last)
        batch_cmr_metrics_2 = batch_ave_steps_cmr_2[best_cmr_idx_2, :, np.arange(pred_2.shape[1])]
        batch_last_metrics_2 = batch_ave_steps_last_2[best_last_idx_2, :, np.arange(pred_2.shape[1])]
        # for i in range(len(metric_scope)):
        eva_cmr_ave.extend(batch_cmr_metrics_2)
        eva_last_ave.extend(batch_last_metrics_2)

    labels = data[4]
    cate_ids = list(map(lambda a: [i[1] for i in a], data[2]))

    print("EPO")
    res = list(evaluate_multi(labels, preds_cmr, cate_ids, metric_scope, isrank, _print))  # [3+2, scope_num, ]
    for j, s in enumerate(params.metric_scope):
        print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f  EVA_AVE: %.4f" % (
            s, res[0][j], res[1][j], res[2][j], res[4][j], res[5][j], np.mean(np.array(eva_cmr_ave), axis=0)[j]))
    # s, res[0][j], res[1][j], res[2][j], res[4][j], 10 * (np.mean(np.array(eva_cmr_ave), axis=0)[j] - 0.9 * res[4][j])))

    print("EVAL TIME: %.4fs" % (time.time() - t))

def reranker_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_time_len', default=10, type=int, help='max time length')
    parser.add_argument('--save_dir', type=str, default='./', help='dir that saves logs and model')
    parser.add_argument('--data_dir', type=str, default='./data/ad/', help='data dir')
    parser.add_argument('--model_type', default='EPO_generator',
                        choices=['PRM', 'DLCM', 'SetRank', 'GSF', 'miDNN', 'Seq2Slate', 'EGR_evaluator',
                                 'EGR_generator', 'CMR_generator', 'LAST_generator','EGR_EPO_generator', 'CMR_EPO_generator','EPO_generator'],
                        type=str,
                        help='algorithm name, including PRM, DLCM, SetRank, GSF, miDNN, Seq2Slate, EGR_evaluator, EGR_generator')
    parser.add_argument('--data_set_name', default='ad', type=str, help='name of dataset, including ad and prm')
    parser.add_argument('--initial_ranker', default='lambdaMART', choices=['DNN', 'lambdaMART'], type=str,
                        help='name of dataset, including DNN, lambdaMART')
    parser.add_argument('--epoch_num', default=30, type=int, help='epochs of each iteration.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--rep_num', default=5, type=int, help='samples repeat number')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2_reg', default=1e-4, type=float, help='l2 loss scale')
    parser.add_argument('--keep_prob', default=0.8, type=float, help='keep probability')
    parser.add_argument('--eb_dim', default=16, type=int, help='size of embedding')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size')
    parser.add_argument('--group_size', default=1, type=int, help='group size for GSF')
    parser.add_argument('--acc_prefer', default=1.0, type=float, help='accuracy_prefer/(accuracy_prefer+diversity)')
    parser.add_argument('--metric_scope', default=[1, 3, 5, 10], type=list, help='the scope of metrics')
    parser.add_argument('--max_norm', default=0, type=float, help='max norm of gradient')
    parser.add_argument('--c_entropy', default=0.001, type=float, help='entropy coefficient in loss')
    # parser.add_argument('--decay_steps', default=3000, type=int, help='learning rate decay steps')
    # parser.add_argument('--decay_rate', default=1.0, type=float, help='learning rate decay rate')
    parser.add_argument('--timestamp', type=str, default=datetime.datetime.now().strftime("%Y%m%d%H%M"))
    parser.add_argument('--evaluator_path', type=str, default='', help='evaluator ckpt dir')
    parser.add_argument('--reload_path', type=str,
                        default='',
                        help='model ckpt dir')
    parser.add_argument('--setting_path', type=str, default='./example/config/ad/epo_generator_setting.json',
                        help='setting dir')
    parser.add_argument('--controllable', type=bool, default=False, help='is controllable')
    parser.add_argument('--evaluator_metrics_path', type=str,
                        # default="./model//save_model_ad/10/202303091111_lambdaMART_LAST_evaluator_16_0.0005_0.0002_64_16_0.8_1.0",
                        default="./model//save_model_ad/10/202502050036_lambdaMART_NS_evaluator_512_0.001_0.0002_64_16_0.8_1.0",
    help='evaluator model')
    parser.add_argument('--with_evaluator_metrics', type=bool, default=True, help='with_evaluator_metrics')
    FLAGS, _ = parser.parse_known_args()
    return FLAGS


if __name__ == '__main__':
    #random.seed(1237)
    #set_global_determinism(1237)
    parse = reranker_parse_args()
    if parse.setting_path:
        parse = load_parse_from_json(parse, parse.setting_path)
    data_set_name = parse.data_set_name
    processed_dir = parse.data_dir
    stat_dir = os.path.join(processed_dir, 'data.stat')
    max_time_len = parse.max_time_len
    initial_ranker = parse.initial_ranker
    if data_set_name == 'prm' and parse.max_time_len > 30:
        max_time_len = 30
    print(parse)
    with open(stat_dir, 'r') as f:
        stat = json.load(f)

    num_item, num_cate, num_ft, profile_fnum, itm_spar_fnum, itm_dens_fnum, = stat['item_num'], stat['cate_num'], \
                                                                              stat['ft_num'], stat['profile_fnum'], \
                                                                              stat['itm_spar_fnum'], stat[
                                                                                  'itm_dens_fnum']
    print('num of item', num_item, 'num of list', stat['train_num'] + stat['val_num'] + stat['test_num'],
          'profile num', profile_fnum, 'spar num', itm_spar_fnum, 'dens num', itm_dens_fnum)

    params = parse
    with open(stat_dir, 'r') as f:
        stat = json.load(f)
    test_dir = os.path.join(processed_dir, initial_ranker + '.data.test')
    if os.path.isfile(test_dir):
        test_lists = pkl.load(open(test_dir, 'rb'))
    else:
        test_lists = construct_list(os.path.join(processed_dir, initial_ranker + '.rankings.test'), max_time_len)
        pkl.dump(test_lists, open(test_dir, 'wb'))

    gpu_options = tf.GPUOptions(allow_growth=True)
    if params.model_type == 'CMR_generator':
        model = CMR_generator(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                              profile_fnum, max_norm=params.max_norm, rep_num=params.rep_num,
                              acc_prefer=params.acc_prefer,
                              is_controllable=params.controllable)
    elif params.model_type == 'SetRank':
        model = SetRank(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                        profile_fnum, max_norm=params.max_norm)
    elif params.model_type == 'DLCM':
        model = DLCM(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                     profile_fnum, max_norm=params.max_norm, acc_prefer=params.acc_prefer)
    elif params.model_type == 'GSF':
        model = GSF(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_fnum, max_norm=params.max_norm, group_size=params.group_size)
    elif params.model_type == 'Seq2Slate':
        model = SLModel(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                        profile_fnum, max_norm=params.max_norm, acc_prefer=params.acc_prefer,
                        is_controllable=params.controllable)
    elif params.model_type == 'EGR_generator' or params.model_type == 'EGR_EPO_generator':
        model = PPOModel(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                         profile_fnum, max_norm=params.max_norm, rep_num=params.rep_num, acc_prefer=params.acc_prefer,
                         is_controllable=params.controllable)
    elif params.model_type == 'PRM':
        model = PRM(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                    profile_fnum, max_norm=params.max_norm, is_controllable=params.controllable,
                    acc_prefer=params.acc_prefer)
    elif params.model_type == 'miDNN':
        model = miDNN(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum, itm_dens_fnum,
                      profile_fnum, max_norm=params.max_norm, is_controllable=params.controllable,
                      acc_prefer=params.acc_prefer)
    elif params.model_type == 'LAST_generator' or params.model_type == 'CMR_EPO_generator':
        model = LAST_generator(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                               itm_dens_fnum, profile_fnum, max_norm=params.max_norm, rep_num=params.rep_num,
                               acc_prefer=1, is_controllable=False)
    elif params.model_type == 'EPO_generator':
        model = EPO(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                         itm_dens_fnum, profile_fnum, max_norm=params.max_norm, rep_num=params.rep_num,
                         acc_prefer=params.acc_prefer, is_controllable=params.controllable,
                         model_name=params.model_type)

    with model.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        model.set_sess(sess)
        sess.run(tf.global_variables_initializer())
        model.load(params.reload_path)

    evaluator = EPO(num_ft, params.eb_dim, params.hidden_size, max_time_len, itm_spar_fnum,
                         itm_dens_fnum, profile_fnum, max_norm=params.max_norm, rep_num=params.rep_num,
                         acc_prefer=params.acc_prefer, is_controllable=params.controllable,
                         model_name=params.model_type)
    with evaluator.graph.as_default() as g:
        sess = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
        evaluator.set_sess(sess)
        sess.run(tf.global_variables_initializer())
        evaluator.load(params.evaluator_metrics_path)

    #loss, res = eval(model, test_lists, params.l2_reg, params.batch_size, True,
    #                 params.metric_scope, with_evaluator=True,
    #                 evaluator=evaluator)
    #print("CMR")
    #for i, s in enumerate(params.metric_scope):
    #    print("@%d  MAP: %.4f  NDCG: %.4f  CLICKS: %.4f  ERR_IA: %.4f  ALPHA_NDCG: %.4f  EVA_AVE: %.4f" % (
    #        s, res[0][i], res[1][i], res[2][i], res[4][i], res[5][i], res[-1][i]))

    #Please remember to change the "sample manner" to the mode you want in the model file, including sampling and greedy
    if params.model_type == 'LAST_generator' or params.model_type == 'CMR_EPO_generator' or params.model_type == 'EGR_PRM_generator' :
        eval_last(model, test_lists, params.batch_size, True, params.metric_scope, evaluator=evaluator)
    elif params.model_type == 'EPO_generator':
        eval_epo(model, test_lists, params.batch_size, True, params.metric_scope, evaluator=evaluator)
