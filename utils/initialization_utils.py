import os
import logging
import json
import torch


def initialize_experiment(params, file_name):
    '''
    Makes the experiment directory, sets standard paths and initializes the logger
    '''
    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')
    exps_dir = os.path.join(params.main_dir, 'experiments')
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)

    params.exp_dir = os.path.join(exps_dir, params.experiment_name)

    if not os.path.exists(params.exp_dir):
        os.makedirs(params.exp_dir)

    if os.path.basename(file_name) == 'test_auc.py':
        params.test_exp_dir = os.path.join(params.exp_dir, f"test_{params.dataset}_{params.constrained_neg_prob}")
        if not os.path.exists(params.test_exp_dir):
            os.makedirs(params.test_exp_dir)
        file_handler = logging.FileHandler(os.path.join(params.test_exp_dir, f"log_test.txt"))
    else:
        file_handler = logging.FileHandler(os.path.join(params.exp_dir, "log_train.txt"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    with open(os.path.join(params.exp_dir, "params.json"), 'w') as fout:
        json.dump(vars(params), fout)


def initialize_model(params, model_class, load_model=False):
    '''
    修正版：采用组装式加载逻辑，确保测试时模型结构与权重完全对齐
    '''
    # 1. 确定权重文件路径
    # 确保文件名与你 Trainer.py 中保存的一致 (如果是 best_model.pth 请手动改下面)
    if load_model and hasattr(params, 'checkpoint_path'):
        checkpoint_path = params.checkpoint_path
    elif getattr(params, 'stage', 1) == 2 and load_model:
        checkpoint_path = os.path.join(params.exp_dir, 'model_stage1.pth')
    else:
        # 🌟 检查点：如果你 Trainer 里存的是 best_model.pth，这里也要改成 best_model.pth
        checkpoint_path = os.path.join(params.exp_dir, 'best_model.pth')
    params.checkpoint_path = checkpoint_path

    # 2. 准备 relation2id (无论加载还是新建都需要)
    relation2id_path = os.path.join(params.main_dir, f'data/{params.dataset}/relation2id.json')
    if not os.path.exists(relation2id_path):
        # 备选路径：防止 main_dir 拼接错误
        relation2id_path = os.path.join(os.getcwd(), f'data/{params.dataset}/relation2id.json')
        
    with open(relation2id_path) as f:
        relation2id = json.load(f)

    # 3. 🌟 核心改进：先在函数顶级作用域初始化模型对象
    # 这样 Pylance 就不会再报“未定义”错误了
    logging.info('Initializing model architecture (creating the skeleton)...')
    
    # 💡 补丁：如果在 initialize_model 里发现漏了参数，最后一次保底补齐
    if not hasattr(params, 'max_label_value') or params.max_label_value is None:
        params.max_label_value = params.hop

    # 构建骨架
    graph_classifier = model_class(params, relation2id).to(device=params.device)

    # 4. 🌟 灌注“记忆”：如果是加载模式且文件存在，则加载权重
    if load_model:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Model checkpoint not found: {checkpoint_path}')

        logging.info('Loading weights from %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=params.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            checkpoint_format = 'checkpoint_dict'
            if checkpoint.get('is_full_prediction_model') is False:
                raise RuntimeError(
                    f"Checkpoint is not a full prediction model: {checkpoint_path}"
                )
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint_format = 'checkpoint_dict'
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            checkpoint_format = 'raw_state_dict'
            state_dict = checkpoint
        else:
            checkpoint_format = 'model_object'
            logging.info('Loaded checkpoint is not a dict; using checkpoint.state_dict().')
            state_dict = checkpoint.state_dict()

        model_state = graph_classifier.state_dict()
        model_keys = set(model_state.keys())
        checkpoint_keys = set(state_dict.keys())
        missing_keys = sorted(model_keys - checkpoint_keys)
        unexpected_keys = sorted(checkpoint_keys - model_keys)
        size_mismatches = sorted([
            key for key in model_keys & checkpoint_keys
            if model_state[key].shape != state_dict[key].shape
        ])

        logging.info("Loaded checkpoint format: %s", checkpoint_format)
        logging.info("Loaded model keys count: %s", len(state_dict))
        logging.info("Missing keys: %s", missing_keys)
        logging.info("Unexpected keys: %s", unexpected_keys)
        if size_mismatches:
            logging.info("Size mismatches: %s", size_mismatches)
            details = [
                f"{key}: model={tuple(model_state[key].shape)} checkpoint={tuple(state_dict[key].shape)}"
                for key in size_mismatches
            ]
            raise RuntimeError("Checkpoint size mismatch: " + "; ".join(details))
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f"Checkpoint key mismatch. Missing keys: {missing_keys}; "
                f"Unexpected keys: {unexpected_keys}"
            )

        graph_classifier.load_state_dict(state_dict, strict=True)
        logging.info("Weights loaded strictly.")

    return graph_classifier


def load_pretrained_vae_branch(params, graph_classifier):
    checkpoint_path = getattr(params, 'load_pretrained_vae', '')
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Pretrained VAE checkpoint not found: {checkpoint_path}')

    logging.info('Loading pretrained VAE/mask branch from %s', checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=params.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        source_state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        source_state = checkpoint['state_dict']
    elif isinstance(checkpoint, dict):
        source_state = checkpoint
    else:
        source_state = checkpoint.state_dict()

    model_state = graph_classifier.state_dict()
    branch_prefixes = ('gnn.encoder.', 'gnn.casual_decoder.')
    expected_keys = sorted([
        key for key in model_state
        if key.startswith(branch_prefixes)
    ])
    branch_state = {
        key: value for key, value in source_state.items()
        if key.startswith(branch_prefixes)
    }

    missing_keys = sorted(set(expected_keys) - set(branch_state.keys()))
    unexpected_branch_keys = sorted([
        key for key in branch_state
        if key not in model_state
    ])
    size_mismatches = sorted([
        key for key in set(expected_keys) & set(branch_state.keys())
        if model_state[key].shape != branch_state[key].shape
    ])
    if missing_keys or unexpected_branch_keys or size_mismatches:
        raise RuntimeError(
            "Pretrained VAE key mismatch. "
            f"Missing keys: {missing_keys}; "
            f"Unexpected branch keys: {unexpected_branch_keys}; "
            f"Size mismatches: {size_mismatches}"
        )

    merged_state = dict(model_state)
    merged_state.update(branch_state)
    graph_classifier.load_state_dict(merged_state, strict=True)

    loaded_tensor_count = len(branch_state)
    loaded_param_count = sum(value.numel() for value in branch_state.values())
    not_loaded_names = sorted([
        key for key in model_state
        if key not in branch_state
    ])
    logging.info(
        "Loaded VAE/mask tensors: %s; scalar parameters: %s",
        loaded_tensor_count,
        loaded_param_count
    )
    logging.info("Parameters not loaded from pretrained VAE: %s", not_loaded_names)

    if getattr(params, 'freeze_vae_after_pretrain', False):
        for parameter in graph_classifier.gnn.encoder.parameters():
            parameter.requires_grad = False
        for parameter in graph_classifier.gnn.casual_decoder.parameters():
            parameter.requires_grad = False
        logging.info("Pretrained VAE/mask branch frozen.")
    else:
        logging.info("Pretrained VAE/mask branch will be finetuned.")
