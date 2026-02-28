# rendering_test 使用说明

本项目当前测试脚本为 `rendering_test.py`，已精简为单文件可运行版本，功能包括：
- 读取点云并渲染参考图
- 加载 `hyper2_comai_sptrans_imgmse111` 对应模型权重进行压缩/解压测试
- 计算性能指标（当前默认 `Y-PSNR`）
- 输出重建渲染图与重建点云文件

## 1. 依赖安装
依赖列表在 `requirement.text`：

```bash
pip install -r requirement.text
```

## 2. 当前测试输入
脚本已固定测试文件：

`/home/hx/pycharm_projet/GPCC_data/long4/longdress.ply`

如需更换，修改 `rendering_test.py` 中的：

```python
filedirs_input = ["/home/hx/pycharm_projet/GPCC_data/long4/longdress.ply"]
```

## 3. 运行方式
在项目根目录执行：

```bash
python rendering_test.py
```

## 4. 输出结果
脚本会在以下目录输出结果：

`imgs_show/compare/longdress/`

主要包含：
- `ref_img_longdress.png`：参考渲染图
- `ai_dec_img_longdress_*.png`：重建渲染图（各 checkpoint）
- `ai_dec_img_longdress_*.ply`：重建点云

终端会打印：
- `bpp` 列表
- `metric` 列表

## 5. 注意事项
- 需要可用的 GPU/CUDA 环境（脚本优先使用 CUDA）。
- 需要项目内可执行文件 `pc_error_d`（用于 `p2p` 指标时）。
- 需要保证 `ckpts_new/sptrans_hyper2_comai_p2p_imgmse111/...` 权重文件存在。下载链接：通过网盘分享的文件：
链接: https://pan.baidu.com/s/1B85azzIVBftUhkOBv4Pbqg?pwd=4v5q 提取码: 4v5q 复制这段内容后打开百度网盘手机App，操作更方便哦
