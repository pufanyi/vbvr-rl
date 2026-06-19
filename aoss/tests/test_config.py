import logging
import os
import unittest

import common_util
from aoss_client.common import exception
from aoss_client.common.config import CONFIG_DEFAULT, Config, Section, _value_to_str

test_dir = os.path.dirname(os.path.realpath(__file__))


class TestSection(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_valuetoStr(self):
        self.assertEqual(_value_to_str(100), "100")
        self.assertEqual(_value_to_str(False), "False")
        expect = {"a": "2", "b": "2", "c": "3", "d": "4"}
        input = dict(a=2, b=2, c=3, d=4)
        self.assertEqual(expect, _value_to_str(input))

    def test_init(self):
        session = Section(CONFIG_DEFAULT)
        self.assertEqual(session._conf, CONFIG_DEFAULT)
        self.assertTrue(isinstance(session, Section))

    def test_key(self):
        session = Section(CONFIG_DEFAULT)
        self.assertEqual(session["boto"], "True")

    def test_ConfigKeyNotFoundError(self):
        session = Section(CONFIG_DEFAULT)
        with self.assertRaises(exception.ConfigKeyNotFoundError):
            _ = session["empty"]

    def test_update(self):
        session = Section(CONFIG_DEFAULT)
        toUpdate = dict(enable_mc="True", file_log_backup_count=3)
        session.update(toUpdate)
        self.assertEqual(session["enable_mc"], "True")
        self.assertEqual(session["file_log_backup_count"], "3")

    # def testGetitem(self):
    # expected = CONFIG_DEFAULT

    def test_get(self):
        session = Section(CONFIG_DEFAULT)
        self.assertEqual(session.get("boto"), "True")

    def test_has_option(self):
        session = Section(CONFIG_DEFAULT)
        self.assertTrue(session.has_option("enable_mem_trace"))
        self.assertFalse(False)

    def test_get_boolean(self):
        session = Section(CONFIG_DEFAULT)
        self.assertFalse(session.get_boolean("enable_mem_trace"))
        with self.assertRaises(exception.ConfigKeyTypeError):
            _ = session.get_boolean("endpoint_url")

    def test_get_int(self):
        session = Section(CONFIG_DEFAULT)
        self.assertEqual(session.get_int("file_log_backup_count"), 1)
        with self.assertRaises(exception.ConfigKeyTypeError):
            _ = session.get_int("boto")

    def test_get_log_level(self):
        session = Section(CONFIG_DEFAULT)
        self.assertEqual(session.get_log_level("file_log_level"), logging.DEBUG)
        with self.assertRaises(exception.ConfigKeyTypeError):
            _ = session.get_log_level("boto")


class TestConfig(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        with self.assertRaises(exception.ConfigFileNotFoundError):
            conf_path = test_dir + "/tests/conf/aoss.conf"
            self.config = Config(conf_path)

        with self.assertRaises(exception.ConfigSectionNotFoundError):
            conf_path = test_dir + "/conf/test_empty.conf"
            self.config = Config(conf_path)

        expect_session = Section(CONFIG_DEFAULT)
        toUpdate = dict(default_cluster="cluster1")
        expect_session.update(toUpdate)

        conf_path = test_dir + "/conf/aoss.conf"
        config = Config(conf_path)
        default_session = config.default()
        self.assertTrue(default_session._conf["boto"] == expect_session._conf["boto"])

        samll_case_conf_path = test_dir + "/conf/test_aoss.conf"
        samll_case_config = Config(samll_case_conf_path)
        samll_case_default_session = samll_case_config.default()
        self.assertTrue(samll_case_default_session._conf["boto"] == expect_session._conf["boto"])

    def test_get(self):
        conf_path = test_dir + "/conf/aoss.conf"
        config = Config(conf_path)
        cluster1_session = config["cluster1"]
        self.assertTrue(cluster1_session.get_boolean("boto"))
        self.assertEqual(cluster1_session.get("access_key"), "lili1")

        samll_case_conf_path = test_dir + "/conf/test_aoss.conf"
        samll_case_config = Config(samll_case_conf_path)
        cluster1_session = samll_case_config["cluster1"]

        self.assertEqual(cluster1_session.get("default_cluster"), "cluster1")

        with self.assertRaises(exception.ConfigSectionNotFoundError):
            config["noncluster1"]

    def test_update(self):
        conf_path = test_dir + "/conf/aoss.conf"
        config = Config(conf_path)
        toUpdate = dict(cluster1=dict(default_cluster="cluster3"))
        config.update(toUpdate)
        self.assertEqual(config["cluster1"].get("default_cluster"), "cluster3")

    def test_items(self):
        conf_path = test_dir + "/conf/aoss.conf"
        config = Config(conf_path)
        sections = config.items()
        self.assertEqual(len(sections), 2)


if __name__ == "__main__":
    common_util.run_test()
