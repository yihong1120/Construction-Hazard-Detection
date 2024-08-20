from __future__ import annotations

import atexit
import unittest
from unittest.mock import patch
from flask import Flask
from flask_jwt_extended import JWTManager
from apscheduler.schedulers.background import BackgroundScheduler
from examples.YOLOv8_server_api.app import app, db, auth_blueprint, detection_blueprint, models_blueprint, update_secret_key

class TestYOLOv8ServerAPI(unittest.TestCase):

    def setUp(self):
        # 每个测试前调用，创建测试客户端
        self.app = app
        self.client = self.app.test_client()
        self.app.testing = True

        # 使用SQLite内存数据库进行测试
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        with self.app.app_context():
            db.create_all()

    def test_app_initialization(self):
        self.assertIsInstance(self.app, Flask)
        self.assertIn('JWT_SECRET_KEY', self.app.config)
        self.assertIsInstance(self.app.config['JWT_SECRET_KEY'], str)
        self.assertGreater(len(self.app.config['JWT_SECRET_KEY']), 0)

    def test_jwt_initialization(self):
        jwt = JWTManager(self.app)
        self.assertIsInstance(jwt, JWTManager)

    def test_blueprints_registration(self):
        self.assertIn(auth_blueprint, self.app.blueprints.values())
        self.assertIn(detection_blueprint, self.app.blueprints.values())
        self.assertIn(models_blueprint, self.app.blueprints.values())

    def test_scheduler_initialization(self):
        scheduler = BackgroundScheduler()
        self.assertIsInstance(scheduler, BackgroundScheduler)

    @patch('examples.YOLOv8_server_api.app.update_secret_key')
    def test_scheduler_job(self, mock_update_secret_key):
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            func=lambda: update_secret_key(self.app),
            trigger='interval',
            days=30,
        )
        jobs = scheduler.get_jobs()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].trigger.interval.days, 30)

    @patch('atexit.register')
    def test_scheduler_shutdown(self, mock_atexit_register):
        scheduler = BackgroundScheduler()
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())
        mock_atexit_register.assert_called_once()

    def tearDown(self):
        # 每个测试后调用，清理数据库
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

if __name__ == '__main__':
    unittest.main()
