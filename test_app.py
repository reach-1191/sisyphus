import unittest
import numpy as np
import cv2
import json
import urllib.parse
from app import app, db, tracker, stream_mgr, User, ExerciseSession, Badge, calculate_angle, calculate_elbow_flare, ema_filter


class ExerciseTrackerTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        tracker.reset_reps()
        tracker.set_exercise("squat")

        with app.app_context():
            # Create a test user for auth testing
            user = User.query.filter_by(username='test_user').first()
            if not user:
                user = User(username='test_user', exp=0, level=1)
                user.set_password('testpass123')
                db.session.add(user)
                db.session.commit()

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

    def test_mode_switch_curl(self):
        # Switch to dumbbell curl mode
        res = self.app.post('/set_exercise', json={'exercise': 'curl'})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(tracker.exercise, "curl")
        self.assertIsNone(tracker.audio_event)
        self.assertTrue(stream_mgr._get_demo_path().endswith("dumbbell_demo.mp4"))

    def test_index_route_unauthenticated(self):
        # Unauthenticated user visiting root '/' redirects directly to '/login'
        response = self.app.get('/')
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.location.endswith('/login'))

    def test_index_route_authenticated(self):
        # Log in test user
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        
        # Authenticated user visiting '/' directly gets the Dashboard
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Level', response.data)
        self.assertIn(b'TOTAL REPS COMPLETED', response.data)
        self.assertIn(b'Badges & Achievements', response.data)
        self.assertIn(b'Daily Rep Volume Over Time', response.data)

    def test_workout_route(self):
        # Visiting '/workout' serves the live workout tracking feed
        response = self.app.get('/workout')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'MediaPipe Tracking Feed', response.data)
        self.assertIn(b'Finish & Save Workout', response.data)

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

    def test_camera_pause_and_resume(self):
        # Test pause endpoint releases camera
        res_pause = self.app.post('/pause_feed')
        self.assertEqual(res_pause.status_code, 200)
        data_pause = json.loads(res_pause.data)
        self.assertTrue(data_pause['paused'])
        self.assertIsNone(stream_mgr.cap)

        # Test resume endpoint
        res_resume = self.app.post('/resume_feed')
        self.assertEqual(res_resume.status_code, 200)
        data_resume = json.loads(res_resume.data)
        self.assertFalse(data_resume['paused'])

    def test_exp_and_badge_saving(self):
        # Log in test user
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})

        # Save workout session
        res = self.app.post(
            '/save_workout',
            json={
                'exercise_type': 'Squats',
                'rep_count': 15,
                'total_duration_seconds': 45.0,
                'avg_form_accuracy': 95.0
            }
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        self.assertGreater(data['exp_gained'], 0)

        with app.app_context():
            u = User.query.filter_by(username='test_user').first()
            self.assertGreater(u.exp, 0)
            # User should have unlocked First Step and Form Perfectionist badges
            badge_names = [b.name for b in u.badges]
            self.assertIn('First Step', badge_names)
            self.assertIn('Form Perfectionist', badge_names)

    def test_audio_alert_endpoint(self):
        # Audio alert endpoint returns json with alert and timestamp
        res = self.app.get('/api/audio_alert')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn('alert', data)
        self.assertIn('timestamp', data)


if __name__ == '__main__':
    unittest.main()
