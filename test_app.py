import unittest
import numpy as np
import cv2
import json
import urllib.parse
from app import app, tracker, calculate_angle, calculate_elbow_flare, ema_filter


class ExerciseTrackerTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        tracker.reset_reps()
        tracker.set_exercise("squat")

    def test_calculate_angle_basic(self):
        hip = [100, 50]
        knee = [100, 100]
        ankle = [100, 150]
        self.assertAlmostEqual(calculate_angle(hip, knee, ankle), 180.0, places=1)

        hip_90 = [50, 100]
        self.assertAlmostEqual(calculate_angle(hip_90, knee, ankle), 90.0, places=1)

    def test_elbow_flare_calculation(self):
        shoulder = [100, 100]
        hip = [100, 200]
        
        elbow_tucked = [150, 150]
        self.assertAlmostEqual(calculate_elbow_flare(shoulder, hip, elbow_tucked), 45.0, places=1)

        elbow_flared = [150, 100]
        self.assertAlmostEqual(calculate_elbow_flare(shoulder, hip, elbow_flared), 90.0, places=1)

    def test_ema_filter(self):
        val1 = ema_filter(100.0, None, alpha=0.35)
        self.assertEqual(val1, 100.0)

        val2 = ema_filter(120.0, val1, alpha=0.35)
        self.assertAlmostEqual(val2, 107.0, places=1)

    def test_silence_on_mode_switch_and_reset(self):
        # Mode switch should be silent
        self.app.post('/set_exercise', json={'exercise': 'pushup'})
        self.assertIsNone(tracker.audio_event)
        self.assertEqual(tracker.vocal_message, "")

        # Reset should be silent
        self.app.post('/reset')
        self.assertIsNone(tracker.audio_event)
        self.assertEqual(tracker.vocal_message, "")

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI Vocal Fitness Coach', response.data)
        self.assertIn(b'Squat deeper!', response.data)
        self.assertIn(b'Keep your knees out!', response.data)
        self.assertIn(b'Lower your chest!', response.data)
        self.assertIn(b'Keep your back straight!', response.data)

    def test_status_route(self):
        response = self.app.get('/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('exercise', data)
        self.assertIn('reps', data)
        self.assertIn('form_score', data)

    def test_process_frame_route(self):
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


if __name__ == '__main__':
    unittest.main()
