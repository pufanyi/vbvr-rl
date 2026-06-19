import os
import unittest
from unittest import mock

import common_util
from aoss_client.client import Client

test_dir = os.path.dirname(os.path.realpath(__file__))


class TestRead(unittest.TestCase):
    def setUp(self):
        # 实例化一个 Mock 对象，用于替换 aoss_client.ceph.s3.s3_client.S3Client.get_with_info 函数
        self._mock_get_with_info = mock.Mock()
        self._mock_get_with_info.return_value = "23", {}
        # 替换 aoss_client.ceph.s3.s3_client.S3Client.get_with_info 函数
        self._patcher = mock.patch("aoss_client.ceph.s3.s3_client.S3Client.get_with_info", self._mock_get_with_info)
        self._patcher.start()
        pass

    def tearDown(self):
        self._patcher.stop()
        pass

    # 1. 在setUp注入Mock方法，替换aoss_client.ceph.s3.s3_client.S3Client.get_with_info函数，并在tearDown停止Mock
    def test_read1(self):
        _conf_path = test_dir + "/conf/aoss.conf"
        c = Client(conf_path=_conf_path)
        # self._mock_get_with_info.return_value = "23", {}
        data = c.get("cluster1:s3://lili1.test2/sometest")
        # print(data)
        self.assertEqual("23", data)

    # 2.使用@mock.patch指定被替换的方法，并默认按参数列表的顺序替换该方法
    @mock.patch("aoss_client.ceph.s3.s3_client.S3Client.get_with_info")
    # @mock.patch("...")
    def test_read2(self, mock_get_with_info):
        mock_get_with_info.return_value = "15", {}
        _conf_path = test_dir + "/conf/aoss.conf"
        c = Client(conf_path=_conf_path)
        data = c.get("cluster1:s3://lili1.test2/sometest")
        self.assertEqual("15", data)

    # 3.使用with mock.patch 替换掉指定模块
    def test_read3(self):
        mock_get_with_info = mock.Mock()
        mock_get_with_info.return_value = "15", {}
        _conf_path = test_dir + "/conf/aoss.conf"
        c = Client(conf_path=_conf_path)
        with mock.patch("aoss_client.ceph.s3.s3_client.S3Client.get_with_info", mock_get_with_info):
            data = c.get("cluster1:s3://lili1.test2/sometest")
            self.assertEqual("15", data)


if __name__ == "__main__":
    common_util.run_test()
