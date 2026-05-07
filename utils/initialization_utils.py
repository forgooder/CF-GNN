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
    if params.stage == 2 and load_model:
        checkpoint_path = os.path.join(params.exp_dir, 'model_stage1.pth')
    else:
        # 🌟 检查点：如果你 Trainer 里存的是 best_model.pth，这里也要改成 best_model.pth
        checkpoint_path = os.path.join(params.exp_dir, 'best_model.pth')

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
        if os.path.exists(checkpoint_path):
            logging.info('Loading weights from %s' % checkpoint_path)
            try:
                # 使用 map_location 确保跨 GPU/CPU 加载不崩溃
                checkpoint = torch.load(checkpoint_path, map_location=params.device)
                
                # 兼容性处理：无论存的是 state_dict 还是整个模型对象都能解开
                if isinstance(checkpoint, dict):
                    graph_classifier.load_state_dict(checkpoint)
                else:
                    # 如果存的是整个模型对象，提取它的权重字典
                    graph_classifier.load_state_dict(checkpoint.state_dict())
                logging.info("✅ Weights loaded successfully.")
            except Exception as e:
                logging.error(f"❌ Error loading weights: {e}")
        else:
            logging.warning(f'⚠️ No existing model found at {checkpoint_path}. Running with random initialization.')

    return graph_classifier
