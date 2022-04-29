import argparse
import ast
import copy
import logging
import multiprocessing
import subprocess
import os
import six
import sys
import time
import numpy as np
import paddle.fluid as fluid
import base_reader
from config import *
import data_rader
from model import transformer
import dist_utils
if os.environ.get('FLAGS_eager_delete_tensor_gb', None) is None:
    os.environ['FLAGS_eager_delete_tensor_gb'] = '0'

num_trainers = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))


def parse_args():
    """
    xun lian de canshu
    :return:
    """
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        "--src_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of source language.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--train_file_pattern",
        type=str,
        required=True,
        help="The pattern to match training data files.")
    parser.add_argument(
        "--val_file_pattern",
        type=str,
        help="The pattern to match validation data files.")
    parser.add_argument(
        "--use_token_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to "
        "produce batch data according to token number.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="The number of sequences contained in a mini-batch, or the maximum "
        "number of tokens (include paddings) contained in a mini-batch. Note "
        "that this represents the number on single device and the actual batch "
        "size for multi-devices will multiply the device number.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=200,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--sort_type",
        default="pool",
        choices=("global", "pool", "none"),
        help="The grain to sort by length: global for all instances; pool for "
        "instances in pool; none for no sort.")
    parser.add_argument(
        "--shuffle",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle instances in each pass.")
    parser.add_argument(
        "--shuffle_batch",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to shuffle the data batches.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        "--token_delimiter",
        type=lambda x: str(x.encode().decode("unicode-escape")),
        default=" ",
        help="The delimiter used to split tokens in source or target sentences. "
        "For EN-DE BPE data we provided, use spaces as token delimiter. ")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--local',
        type=ast.literal_eval,
        default=True,
        help='Whether to run as local mode.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help="The device type.")
    parser.add_argument(
        '--update_method',
        choices=("pserver", "nccl2"),
        default="pserver",
        help='Update method.')
    parser.add_argument(
        '--sync', type=ast.literal_eval, default=True, help="sync mode.")
    parser.add_argument(
        "--enable_ce",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to run the task "
        "for continuous evaluation.")
    parser.add_argument(
        "--use_mem_opt",
        type=ast.literal_eval,
        default=True,
        help="The flag indicating whether to use memory optimization.")
    parser.add_argument(
        "--use_py_reader",
        type=ast.literal_eval,
        default=False,
        help="The flag indicating whether to use py_reader.")
    parser.add_argument(
        "--fetch_steps",
        type=int,
        default=100,
        help="The frequency to fetch and print output.")

    args = parser.parse_args()
    # Append args related to dict
    src_dict = base_reader.DataReader.load_dict(args.src_vocab_fpath)
    trg_dict = base_reader.DataReader.load_dict(args.trg_vocab_fpath)
    dict_args = [
        "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
        str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
        "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
        str(src_dict[args.special_token[2]])
    ]
    merge_cfg_from_list(args.opts + dict_args,
                        [TrainTaskConfig, ModelHyperParams])
    return args


def get_device_num():
    # NOTE(zcd): for multi-processe training, each process use one GPU card.
    if num_trainers > 1: return 1
    visible_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    if visible_device:
        device_num = len(visible_device.split(','))
    else:
        device_num = subprocess.check_output(
            ['nvidia-smi', '-L']).decode().count('\n')
    return device_num


def append_nccl2_prepare(startup_prog, trainer_id, worker_endpoints,
                         current_endpoint):
    assert (trainer_id >= 0 and len(worker_endpoints) > 1 and
            current_endpoint in worker_endpoints)
    eps = copy.deepcopy(worker_endpoints)
    eps.remove(current_endpoint)
    nccl_id_var = startup_prog.global_block().create_var(
        name="NCCLID", persistable=True, type=fluid.core.VarDesc.VarType.RAW)
    startup_prog.global_block().append_op(
        type="gen_nccl_id",
        inputs={},
        outputs={"NCCLID": nccl_id_var},
        attrs={
            "endpoint": current_endpoint,
            "endpoint_list": eps,
            "trainer_id": trainer_id
        })
    return nccl_id_var


def test_context(exe, train_exe, dev_count):
    # Context to do validation.
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    if args.enable_ce:
        test_prog.random_seed = 1000
        startup_prog.random_seed = 1000
    with fluid.program_guard(test_prog, startup_prog):
        with fluid.unique_name.guard():
            sum_cost, avg_cost, predict, token_num, pyreader = transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                TrainTaskConfig.label_smooth_eps,
                use_py_reader=args.use_py_reader,
                is_test=True)
    test_prog = test_prog.clone(for_test=True)
    test_data, _ = data_rader.prepare_data_generator(
        args,
        is_test=True,
        count=dev_count,
        pyreader=pyreader,
        py_reader_provider_wrapper=data_rader.py_reader_provider_wrapper)

    exe.run(startup_prog)  # to init pyreader for testing
    if TrainTaskConfig.ckpt_path:
        fluid.io.load_persistables(
            exe, TrainTaskConfig.ckpt_path, main_program=test_prog)

    build_strategy = fluid.BuildStrategy()
    test_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        main_program=test_prog,
        build_strategy=build_strategy,
        share_vars_from=train_exe)

    def test(exe=test_exe, pyreader=pyreader):
        test_total_cost = 0
        test_total_token = 0

        if args.use_py_reader:
            pyreader.start()
            data_generator = None
        else:
            data_generator = test_data()
        while True:
            try:
                feed_dict_list = data_rader.prepare_feed_dict_list(data_generator, False,
                                                        dev_count)
                outs = exe.run(fetch_list=[sum_cost.name, token_num.name],
                                    feed=feed_dict_list)
            except (StopIteration, fluid.core.EOFException):
                # The current pass is over.
                if args.use_py_reader:
                    pyreader.reset()
                break
            sum_cost_val, token_num_val = np.array(outs[0]), np.array(outs[1])
            test_total_cost += sum_cost_val.sum()
            test_total_token += token_num_val.sum()
        test_avg_cost = test_total_cost / test_total_token
        test_ppl = np.exp([min(test_avg_cost, 100)])
        return test_avg_cost, test_ppl

    return test


def train_loop(exe,
               train_prog,
               startup_prog,
               dev_count,
               sum_cost,
               avg_cost,
               token_num,
               pyreader,
               nccl2_num_trainers=1,
               nccl2_trainer_id=0):
    # Initialize the parameters.
    if TrainTaskConfig.ckpt_path:
        exe.run(startup_prog)  # to init pyreader for training
        logging.info("load checkpoint from {}".format(
            TrainTaskConfig.ckpt_path))
        fluid.io.load_persistables(
            exe, TrainTaskConfig.ckpt_path, main_program=train_prog)
    else:
        logging.info("init fluid.framework.default_startup_program")
        exe.run(startup_prog)

    logging.info("begin reader")
    train_data, _ = data_rader.prepare_data_generator(
        args,
        is_test=False,
        count=dev_count,
        pyreader=pyreader,
        py_reader_provider_wrapper=data_rader.py_reader_provider_wrapper)

    # For faster executor
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = int(args.fetch_steps)
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = True

    sum_cost.persistable = True
    token_num.persistable = True
    # Since the token number differs among devices, customize gradient scale to
    # use token average cost among multi-devices. and the gradient scale is
    # `1 / token_number` for average cost.
    # build_strategy.gradient_scale_strategy = fluid.BuildStrategy.GradientScaleStrategy.Customized
    build_strategy.fuse_all_optimizer_ops = True

    if num_trainers > 1 and args.use_py_reader and TrainTaskConfig.use_gpu:
        dist_utils.prepare_for_multi_process(exe, build_strategy, train_prog)
        exec_strategy.num_threads = 1

    logging.info("begin executor")
    train_exe = fluid.ParallelExecutor(
        use_cuda=TrainTaskConfig.use_gpu,
        loss_name=avg_cost.name,
        main_program=train_prog,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy,
        num_trainers=nccl2_num_trainers,
        trainer_id=nccl2_trainer_id)

    if args.val_file_pattern is not None:
        test = test_context(exe, train_exe, dev_count)

    # the best cross-entropy value with label smoothing
    loss_normalizer = -((1. - TrainTaskConfig.label_smooth_eps) * np.log(
        (1. - TrainTaskConfig.label_smooth_eps
         )) + TrainTaskConfig.label_smooth_eps *
                        np.log(TrainTaskConfig.label_smooth_eps / (
                            ModelHyperParams.trg_vocab_size - 1) + 1e-20))

    step_idx = 0
    init_flag = True
    logging.info("begin train")
    print(dev_count)
    for pass_id in six.moves.xrange(TrainTaskConfig.pass_num):
        pass_start_time = time.time()

        if args.use_py_reader:
            pyreader.start()
            data_generator = None
        else:
            data_generator = train_data()

        batch_id = 0
        while True:
            try:
                feed_dict_list = data_rader.prepare_feed_dict_list(data_generator,
                                                                   init_flag, dev_count)
                outs = train_exe.run(
                    fetch_list=[sum_cost.name, token_num.name]
                    if step_idx % args.fetch_steps == 0 else [],
                    feed=feed_dict_list)

                if step_idx % args.fetch_steps == 0:
                    sum_cost_val, token_num_val = np.array(outs[0]), np.array(
                        outs[1])
                    # sum the cost from multi-devices
                    total_sum_cost = sum_cost_val.sum()
                    total_token_num = token_num_val.sum()
                    total_avg_cost = total_sum_cost / total_token_num

                    if step_idx == 0:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)])))
                        avg_batch_time = time.time()
                    else:
                        logging.info(
                            "step_idx: %d, epoch: %d, batch: %d, avg loss: %f, "
                            "normalized loss: %f, ppl: %f, speed: %.2f step/s" %
                            (step_idx, pass_id, batch_id, total_avg_cost,
                             total_avg_cost - loss_normalizer,
                             np.exp([min(total_avg_cost, 100)]),
                             args.fetch_steps / (time.time() - avg_batch_time)))
                        avg_batch_time = time.time()

                if step_idx % TrainTaskConfig.save_freq == 0 and step_idx > 0:
                    fluid.io.save_persistables(
                        exe,
                        os.path.join(TrainTaskConfig.ckpt_dir,
                                     "latest.checkpoint"), train_prog)


                init_flag = False
                batch_id += 1
                step_idx += 1
            except (StopIteration, fluid.core.EOFException):
                # The current pass is over.
                if args.use_py_reader:
                    pyreader.reset()
                break

        time_consumed = time.time() - pass_start_time
        # Validate and save the persistable.
        if args.val_file_pattern is not None:
            val_avg_cost, val_ppl = test()
            logging.info(
                "epoch: %d, val avg loss: %f, val normalized loss: %f, val ppl: %f,"
                " consumed %fs" % (pass_id, val_avg_cost,
                                   val_avg_cost - loss_normalizer, val_ppl,
                                   time_consumed))
        else:
            logging.info("epoch: %d, consumed %fs" % (pass_id, time_consumed))

        if not args.enable_ce:
            fluid.io.save_persistables(
                exe,
                os.path.join(TrainTaskConfig.ckpt_dir,
                             "pass_" + str(pass_id) + ".checkpoint"),
                train_prog)

    if args.enable_ce:  # For CE
        print("kpis\ttrain_cost_card%d\t%f" % (dev_count, total_avg_cost))
        if args.val_file_pattern is not None:
            print("kpis\ttest_cost_card%d\t%f" % (dev_count, val_avg_cost))
        print("kpis\ttrain_duration_card%d\t%f" % (dev_count, time_consumed))


def train(args):
    # priority: ENV > args > config
    is_local = os.getenv("PADDLE_IS_LOCAL", "1")
    if is_local == '0':
        args.local = False
    logging.info(args)

    if args.device == 'CPU':
        TrainTaskConfig.use_gpu = False

    training_role = os.getenv("TRAINING_ROLE", "TRAINER")

    if training_role == "PSERVER" or (not TrainTaskConfig.use_gpu):
        place = fluid.CPUPlace()
        dev_count = int(os.environ.get('CPU_NUM', multiprocessing.cpu_count()))
    else:
        gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
        place = fluid.CUDAPlace(gpu_id)
        dev_count = get_device_num()

    exe = fluid.Executor(place)

    train_prog = fluid.Program()
    startup_prog = fluid.Program()

    if args.enable_ce:
        train_prog.random_seed = 1000
        startup_prog.random_seed = 1000

    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            sum_cost, avg_cost, predict, token_num, pyreader = transformer(
                ModelHyperParams.src_vocab_size,
                ModelHyperParams.trg_vocab_size,
                ModelHyperParams.max_length + 1,
                ModelHyperParams.n_layer,
                ModelHyperParams.n_head,
                ModelHyperParams.d_key,
                ModelHyperParams.d_value,
                ModelHyperParams.d_model,
                ModelHyperParams.d_inner_hid,
                ModelHyperParams.prepostprocess_dropout,
                ModelHyperParams.attention_dropout,
                ModelHyperParams.relu_dropout,
                ModelHyperParams.preprocess_cmd,
                ModelHyperParams.postprocess_cmd,
                ModelHyperParams.weight_sharing,
                TrainTaskConfig.label_smooth_eps,
                ModelHyperParams.bos_idx,
                use_py_reader=args.use_py_reader,
                is_test=False)

            if args.sync:
                lr_decay = fluid.layers.learning_rate_scheduler.noam_decay(
                    ModelHyperParams.d_model, TrainTaskConfig.warmup_steps)
                logging.info("before adam")

                with fluid.default_main_program()._lr_schedule_guard():
                    learning_rate = lr_decay * TrainTaskConfig.learning_rate

                optimizer = fluid.optimizer.Adam(
                    learning_rate=learning_rate,
                    beta1=TrainTaskConfig.beta1,
                    beta2=TrainTaskConfig.beta2,
                    epsilon=TrainTaskConfig.eps)
            else:
                optimizer = fluid.optimizer.SGD(0.003)
            optimizer.minimize(avg_cost)

    if args.local:
        logging.info("local start_up:")
        print(dev_count)
        train_loop(exe, train_prog, startup_prog, dev_count, sum_cost, avg_cost,
                   token_num, pyreader)
    else:
        if args.update_method == "nccl2":
            trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
            port = os.getenv("PADDLE_PORT")
            worker_ips = os.getenv("PADDLE_TRAINERS")
            worker_endpoints = []
            for ip in worker_ips.split(","):
                worker_endpoints.append(':'.join([ip, port]))
            trainers_num = len(worker_endpoints)
            current_endpoint = os.getenv("POD_IP") + ":" + port
            if trainer_id == 0:
                logging.info("train_id == 0, sleep 60s")
                time.sleep(60)
            logging.info("trainers_num:{}".format(trainers_num))
            logging.info("worker_endpoints:{}".format(worker_endpoints))
            logging.info("current_endpoint:{}".format(current_endpoint))
            append_nccl2_prepare(startup_prog, trainer_id, worker_endpoints,
                                 current_endpoint)
            train_loop(exe, train_prog, startup_prog, dev_count, sum_cost,
                       avg_cost, token_num, pyreader, trainers_num,
                       trainer_id)
            return

        port = os.getenv("PADDLE_PORT", "6174")
        pserver_ips = os.getenv("PADDLE_PSERVERS")  # ip,ip...
        eplist = []
        for ip in pserver_ips.split(","):
            eplist.append(':'.join([ip, port]))
        pserver_endpoints = ",".join(eplist)  # ip:port,ip:port...
        trainers = int(os.getenv("PADDLE_TRAINERS_NUM", "0"))
        current_endpoint = os.getenv("POD_IP") + ":" + port
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))

        logging.info("pserver_endpoints:{}".format(pserver_endpoints))
        logging.info("current_endpoint:{}".format(current_endpoint))
        logging.info("trainer_id:{}".format(trainer_id))
        logging.info("pserver_ips:{}".format(pserver_ips))
        logging.info("port:{}".format(port))

        t = fluid.DistributeTranspiler()
        t.transpile(
            trainer_id,
            pservers=pserver_endpoints,
            trainers=trainers,
            program=train_prog,
            startup_program=startup_prog)

        if training_role == "PSERVER":
            logging.info("distributed: pserver started")
            current_endpoint = os.getenv("POD_IP") + ":" + os.getenv(
                "PADDLE_PORT")
            if not current_endpoint:
                logging.critical("need env SERVER_ENDPOINT")
                exit(1)
            pserver_prog = t.get_pserver_program(current_endpoint)
            pserver_startup = t.get_startup_program(current_endpoint,
                                                    pserver_prog)

            exe.run(pserver_startup)
            exe.run(pserver_prog)
        elif training_role == "TRAINER":
            logging.info("distributed: trainer started")
            train_loop(exe, train_prog, startup_prog, dev_count, sum_cost,
                       avg_cost, predict, pyreader)
        else:
            logging.critical(
                "environment var TRAINER_ROLE should be TRAINER os PSERVER")
            exit(1)


if __name__ == "__main__":
    LOG_FORMAT = "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT)
    logging.getLogger().setLevel(logging.INFO)

    args = parse_args()
    train(args)