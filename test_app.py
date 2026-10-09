import unittest
import numpy as np
import cv2
import json
import urllib.parse
from app import app, tracker, calculate_angle


class SquatTrackerTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        tracker.reset()

    def test_calculate_angle_straight_line(self):
        # 180 degree line: hip (100, 50), knee (100, 100), ankle (100, 150)
        hip = [100, 50]
        knee = [100, 100]
        ankle = [100, 150]
        angle = calculate_angle(hip, knee, ankle)
        self.assertAlmostEqual(angle, 180.0, places=1)

    def test_calculate_angle_right_angle(self):
        # 90 degree corner: hip (50, 100), knee (100, 100), ankle (100, 150)
        hip = [50, 100]
        knee = [100, 100]
        ankle = [100, 150]
        angle = calculate_angle(hip, knee, ankle)
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_calculate_angle_acute(self):
        # 45 degree angle: vector (50, 50) and vector (0, 50)
        hip = [100 + 50, 100 + 50]
        knee = [100, 100]
        ankle = [100, 150]
        angle = calculate_angle(hip, knee, ankle)
        self.assertAlmostEqual(angle, 45.0, places=1)

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI Squat Tracker', response.data)
        self.assertIn(b'Use Browser Camera', response.data)

    def test_status_route(self):
        response = self.app.get('/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('reps', data)
        self.assertIn('stage', data)
        self.assertIn('angle', data)
        self.assertIn('feedback', data)

    def test_reset_route(self):
        tracker.counter = 5
        response = self.app.post('/reset')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tracker.counter, 0)

    def test_set_thresholds_route(self):
        response = self.app.post('/set_thresholds', json={
            'down_threshold': 90,
            'up_threshold': 160
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tracker.down_threshold, 90.0)
        self.assertEqual(tracker.up_threshold, 160.0)

    def test_process_frame_browser_camera(self):
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', frame)
        
        response = self.app.post(
            '/process_frame',
            data=buf.tobytes(),
            content_type='image/jpeg'
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'image/jpeg')
        self.assertIn('X-Metrics', response.headers)
        
        metrics_raw = urllib.parse.unquote(response.headers['X-Metrics'])
        metrics = json.loads(metrics_raw)
        self.assertIn('reps', metrics)
        self.assertIn('stage', metrics)


if __name__ == '__main__':
    unittest.main()
