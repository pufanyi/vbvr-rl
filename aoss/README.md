AOSS Python SDK 2.0
===

注意：该版本SDK需要python3.6环境

若之前安装过旧版本，请先运行

```bash
$ pip uninstall pycephs3client
$ rm -rf ~/.local/lib/python3.6/site-packages/aoss_client
```

## 建议在安装之前先升级 pip

```bash
source /mnt/lustre/share/platform/env/<pat... or pt...> # 请根据实际情况确定是否需要 source
python3 -m pip install --upgrade pip # 请根据实际情况确定是否需要 `sudo` 或添加 `--user` 参数
```

## 训练集群环境上安装

```bash
$ source /mnt/lustre/share/platform/env/<pat... or pt...>
$ python setup.py sdist
$ pip install --user dist/*
```

## 通过修改 PYTHONPATH 安装

```bash
$ source /mnt/lustre/share/platform/env/<pat... or pt...>

# 安装SDK依赖
$ python setup.py egg_info
$ pip install -r *.egg-info/requires.txt

# 将SDK编译到 ./build 目录
$ python setup.py build

# 修改 PYTHONPATH 环境变量
$ export PYTHONPATH=<path_to_sdk>/build/lib:$PYTHONPATH
```

## venv环境上安装

```bash
$ python3 -m venv your_venv_name # 若已创建venv环境则无需执行
$ source your_venv_name/bin/active
$ python setup.py sdist
$ pip install dist/*
```

## 系统环境上安装

```bash
$ python3 setup.py sdist
$ python3 -m pip install dist/* # 请根据实际情况确定是否需要 `sudo` 或添加 `--user` 参数
```

## 使用

SDK 提供 `get`、`put`、`upload_fileobj`、`upload_file`、`download_file`、`download_fileobj` 接口，使用方式为：

```python
data = client.get(url)
```

```python
client.put(url, data)
```

```python
# `upload_fileobj`、`upload_file`接受一个可选的callback回调参数。
with open(filename, 'rb') as data:           # 二进制分片上传
    client.upload_fileobj(uri, data, s3back)

client.upload_file(uri, filename)            # 大文件分片上传
```

```python
# `download_file`、`download_fileobj`接受一个可选的callback回调参数。
with open(filename, 'wb') as data:             # 二进制分片下载
    client.download_fileobj(uri, data, s3back)

client.download_file(uri, filename, s3back)    # 大文件下载
```

``注意：`` `upload_fileobj`、`upload_file`、`download_file`、`download_fileobj` 接口需要在 `aoss.conf` 设置 `multipart_threshold`、`max_concurrency`、`multipart_chunksize`参数

以下为使用 SDK 读取图片、进行图片处理后并保存图片的简单例子

```python
import cv2
import numpy as np
from os.path import splitext
from aoss_client.client import Client

conf_path = '~/aoss.conf'
client = Client(conf_path) # 若不指定 conf_path ，则从 '~/aoss.conf' 读取配置文件
img_url = 's3://bucket1/image.jpeg'
img_gray_url = 's3://bucket1/image_gray.jpeg'
img_ext = splitext(img_gray_url)[-1]

# 图片读取
img_bytes = client.get(img_url)
assert(img_bytes is not None)
img_mem_view = memoryview(img_bytes)
img_array = np.frombuffer(img_mem_view, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# 图片处理
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 图片存储
success, img_gray_array = cv2.imencode(img_ext, img_gray)
assert(success)
img_gray_bytes = img_gray_array.tostring()
client.put(img_gray_url, img_gray_bytes)
```

配置文件请参考 [aoss.conf](./conf/aoss.conf)

``请注意：配置文件中 `key = value` 的 key 前面不能有空格，否则该行视为上一行配置项 value 的一部分``

使用样例请参考 [multi_cluster_test.py](./tests/multi_cluster_test.py)

## `upload_file`、`upload_fileobj`、`download_file`、`download_fileobj` 大文件上传、下载
使用样例 [upload_download_file_test.py](./tests/upload_download_file_test.py)

## `Tensor` 和 `Json` 数据保存与读取
使用样例 [tensor_json_test.py](./tests/tensor_json_test.py)

## `Pillow Image` 数据保存与读取
使用样例 [pillow_image_test.py](./tests/pillow_image_test.py)

## 数据过大无法上传，则需要分片上传
使用样例 [multipart_test.py](./tests/multipart_test.py)

## 创建 Bucket
```python
client.create_bucket('s3://mybucket')
```

## 顺序的读取某个前缀的数据
```python
cluster = 'cluster1'
files = client.get_file_iterator('cluster1:s3://lili1.test2/test3')
for p, k in files:
    key = '{0}:s3://{1}'.format(cluster, p)
    data = client.get(key)
```
## 使用 anonymous 账户访问数据
若在配置文件中不设置 `access_key` 和 `secret_key`，将以 `anonymous` 账户访问数据。


## 使用伪客户端

在对应客户端添加如下配置:

```conf
fake = True
```

配置文件请参考 [fake_client.conf](./conf/fake_client.conf)

使用样例请参考 [fake_client_test.py](./tests/fake_client_test.py)


## IO 统计信息

IO 统计信息可通过以下三种方式修改其`log`输出频度：
- 由环境变量 `count_disp` 设置
- 由配置文件 `count_disp` 设置 （若已设置环境变量，则该方式无效）
- 调用 `client.set_count_disp(count_disp)` (该方式将覆盖上述两种方式），但限于`parrots`和`pytorch`的运行机制，在某些使用场景下可能无法有效修改。

若 `count_disp` 为 `0` ，则将关闭 IO 统计信息打印。

若需要在 `console` 中打印 IO 统计信息，则需要设置 `console_log_level` 为 `INFO` 或更低级别，且 `count_disp` 需大于 `0`。


## DataLoader

`SDK` 提供的 `DataLoader` 额外支持如下参数：

- `prefetch_factor`，默认2。每个 `worker` 预读 `batch` 数目。
- `persistent_workers`，默认 `False`。如果为 `True`，则每轮 `epoch` 迭代完毕后 `worker` 进程将不会关闭，下轮 `epoch` 将复用该 `worker` 进程。

用例：

```python
from aoss_client.utils.data import DataLoader
dataloader = DataLoader(dataset=xxx, ..., prefetch_factor=4, persistent_workers=True)
```

## SSL 验证

使用 `https` 协议时默认不会对 `SSL` 进行验证。若需要开启验证，请在配置文件中进行如下设置
```conf
verify_ssl = True
```

## Presigned URL，生成签名链接

```python
presigned_url = client.generate_presigned_url(url, client_method ='get_object', expires_in=3600)
```

`client_method` 取值为 `get_object` (默认值) 或 `put_object`

`expires_in` 单位为秒，默认值为 3600

## Presigned POST，生成签名 POST

```python
presigned_post = client.generate_presigned_post(url, fields=None, conditions=None, expires_in=3600)
```

参数及返回值详见 [generate_presigned_post](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.generate_presigned_post)，其中参数 bucket 和 key 从 url 中提取。

## Range 读取
range 的形式有如下几种
- '`first_byte_pos`-'：读取`first_byte_pos`及后续的所有数据
- '`first_byte_pos`-`last_byte_pos`'：读取`first_byte_pos`到`last_byte_pos`（包含）的数据
- '-`suffix_length`'：读取最后`suffix_length`字节的数据

具体详见 HTTP 中 Range 的定义：https://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.35 ，但ceph目前不支持多个 Range 读取。

```python
# 假设数据长度为 10000 字节

# 读取第一段 500 字节数据
data1 = client.get(url, range='0-499')

# 读取第二段 500 字节数据
data2 = client.get(url, range='500-999')

# 读取最后 500 字节数据
data_last = client.get(url, range='-500')
# 或
data_last = client.get(url, range='9500-')

```
## 以流的形式读取数据
```python
stream = client.get(url, enable_stream=True)
```
返回的 `stream` 为 `StreamingBody`，使用方法详见
https://botocore.amazonaws.com/v1/documentation/api/latest/reference/response.html

## 判断对象是否存在
```python
exists = client.contains(url)
```

## 查询对象大小
```python
size = client.size(url)
```
若对象不存在，产生 `NoSuchKeyError` 异常

## 删除对象
```python
client.delete(url)
```

## 列出当前路径包含的对象或目录
```python
contents = client.list(url)
for content in contents:
    if content.endswith('/'):
        print('directory:', content)
    else:
        print('object:', content)
```

## 判断目录是否存在
```python
client.isdir(url)
```

注意：`Ceph`中没有目录的概念，本函数返回`True`时代表存在以该`url`作为前缀的对象，其他情况返回`False`。


## 使用 `/mnt/cache` 目录下的 `Python` 环境
相对于 `/mnt/lustre` 目录，在 `/mnt/cache` 目录执行 `Python` 有一定的性能提升。
使用方式如下:
- `source` `/mnt/cache` 目录下的 `Python` 环境
```bash
### 例如 pt1.3v1
source /mnt/cache/share/platform/env/pt1.3v1
### 或 s0.3.3
source /mnt/cache/share/spring/s0.3.3
```

- 检查 `Python` 路径是否正确
```bash
which python
### 结果应为 /mnt/cache/...
```

- 设定 `PYTHONUSERBASE` 环境变量
```bash
export PYTHONUSERBASE=/mnt/cache/<username>/.local
```

- 重新安装相关依赖库（仅需首次使用时执行）
```
python -m pip install --user <packages>
```
