#!/bin/bash

# 1. 先杀掉旧的 tensorboard 进程（防止报错端口被占用）
echo "🔪 清理旧的 TensorBoard 进程..."
pkill -f tensorboard

# 2. 后台启动新的
echo "启动 TensorBoard..."
# --bind_all 允许 SSH 隧道访问，日志输出扔进黑洞(/dev/null)保持清爽
nohup tensorboard --logdir=./logs --port 6006 --bind_all > /dev/null 2>&1 &

echo "TensorBoard 已在后台运行！"
echo "请在本地浏览器访问: http://localhost:6006"
echo "(前提：你已经建立了 SSH 隧道或者 VS Code 端口转发)"