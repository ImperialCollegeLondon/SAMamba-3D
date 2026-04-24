class Config:
    """训练配置"""
    # 数据路径
    data_path = None
    labels_path = None

    # Patch参数
    patch_size = (96,96,96)
    stride = tuple(int(p * (1 - 0.5)) for p in patch_size)
    # patch_size = (32,32,32)
    # stride = (16, 16, 16)
    num_classes = 4

    # 训练参数
    batch_size = 1
    num_epochs = 200
    learning_rate = 1e-4
    weight_decay = 1e-5


    # 数据划分
    train_patches = 256 
    val_patches = 256

    # 其他
    num_workers = 4
    save_dir = None
    log_dir = None
    result_dir = None
    seed = 42