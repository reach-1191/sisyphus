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

    def test_adaptive_welcome_messaging_new_user(self):
        with app.app_context():
            new_u = User.query.filter_by(username='new_athlete').first()
            if not new_u:
                new_u = User(username='new_athlete')
                new_u.set_password('pass123')
                db.session.add(new_u)
                db.session.commit()

        self.app.post('/login', data={'username': 'new_athlete', 'password': 'pass123'})
        res = self.app.get('/dashboard')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Welcome to Sisyphus, new_athlete! Ready to start your fitness journey?', res.data)
        self.assertIn(b'Quick Onboarding Guide', res.data)

    def test_adaptive_welcome_messaging_returning_user(self):
        # test_user has logged workout sessions from test_exp_and_badge_saving
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        res = self.app.get('/dashboard')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Welcome back, test_user! Ready to crush today', res.data)
        self.assertIn(b'Streak', res.data)

    def test_visual_exercise_choice_cards(self):
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        res = self.app.get('/dashboard')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Glutes / Legs', res.data)
        self.assertIn(b'Chest / Core', res.data)
        self.assertIn(b'Biceps', res.data)
        self.assertIn(b'Squats', res.data)
        self.assertIn(b'Push-ups', res.data)
        self.assertIn(b'Dumbbell Curls', res.data)

    def test_summary_screen_and_form_review(self):
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        # Save a session
        res = self.app.post(
            '/save_workout',
            json={
                'exercise_type': 'Squats',
                'rep_count': 10,
                'total_duration_seconds': 30.0,
                'avg_form_accuracy': 88.0
            }
        )
        data = json.loads(res.data)
        session_id = data['session_id']

        # Visit summary screen
        res_summary = self.app.get(f'/summary/{session_id}')
        self.assertEqual(res_summary.status_code, 200)
        self.assertIn(b'Workout Completed!', res_summary.data)
        self.assertIn(b'Movement Precision & Coach Feedback', res_summary.data)
        self.assertIn(b'Optimal', res_summary.data)
        self.assertIn(b'Squat depth consistent on 88% of reps', res_summary.data)
        self.assertIn(b'Total Reps', res_summary.data)
        self.assertIn(b'Session Duration', res_summary.data)
        self.assertIn(b'Form Score', res_summary.data)
        self.assertIn(b'EXP Earned', res_summary.data)
        # Verify Coach Insights Bento Card
        self.assertIn(b'Coach Insights', res_summary.data)
        self.assertIn(b'Positive Encouragement', res_summary.data)
        self.assertIn(b'Technical Improvement Tip', res_summary.data)
        self.assertIn(b'Squats: Drive knees outward over toes throughout descent.', res_summary.data)

    def test_coach_insights_precision_over_90(self):
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        res = self.app.post(
            '/save_workout',
            json={
                'exercise_type': 'Squats',
                'rep_count': 15,
                'total_duration_seconds': 45.0,
                'avg_form_accuracy': 95.0
            }
        )
        session_id = json.loads(res.data)['session_id']
        res_summary = self.app.get(f'/summary/{session_id}')
        self.assertEqual(res_summary.status_code, 200)
        self.assertIn(b'Precision: Flawless movement pattern. Maintain this rhythm next set.', res_summary.data)
        self.assertIn(b'Outstanding effort!', res_summary.data)

    def test_coach_insights_pushups_and_curls_tips(self):
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        # Push-ups with score < 75
        res_push = self.app.post(
            '/save_workout',
            json={
                'exercise_type': 'Push-Ups',
                'rep_count': 8,
                'total_duration_seconds': 25.0,
                'avg_form_accuracy': 65.0
            }
        )
        sid_push = json.loads(res_push.data)['session_id']
        res_summary_push = self.app.get(f'/summary/{sid_push}')
        self.assertIn(b'Push-ups: Keep spine neutral from neck to ankles.', res_summary_push.data)

        # Curls with score < 75
        res_curl = self.app.post(
            '/save_workout',
            json={
                'exercise_type': 'Dumbbell Curls',
                'rep_count': 10,
                'total_duration_seconds': 30.0,
                'avg_form_accuracy': 70.0
            }
        )
        sid_curl = json.loads(res_curl.data)['session_id']
        res_summary_curl = self.app.get(f'/summary/{sid_curl}')
        self.assertIn(b'Curls: Pin elbows against your ribs.', res_summary_curl.data)

    def test_coach_insights_tone_and_word_count_rules(self):
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        prohibited_phrases = ["not bad", "not terrible", "not uncommon", "no small feat", "not great", "not poor"]
        
        test_cases = [
            ('Squats', 95.0),
            ('Squats', 85.0),
            ('Squats', 65.0),
            ('Push-Ups', 92.0),
            ('Push-Ups', 80.0),
            ('Push-Ups', 60.0),
            ('Dumbbell Curls', 96.0),
            ('Dumbbell Curls', 78.0),
            ('Dumbbell Curls', 55.0),
        ]
        
        for exercise, score in test_cases:
            res = self.app.post(
                '/save_workout',
                json={
                    'exercise_type': exercise,
                    'rep_count': 10,
                    'total_duration_seconds': 30.0,
                    'avg_form_accuracy': score
                }
            )
            sid = json.loads(res.data)['session_id']
            res_summary = self.app.get(f'/summary/{sid}')
            html_text = res_summary.data.decode('utf-8')
            
            # Extract Coach Insights section
            self.assertIn('Coach Insights', html_text)
            self.assertIn('Positive Encouragement', html_text)
            self.assertIn('Technical Improvement Tip', html_text)
            
            # Check litotes prohibition
            lower_html = html_text.lower()
            for forbidden in prohibited_phrases:
                self.assertNotIn(forbidden, lower_html)

    def test_save_workout_computed_form_accuracy(self):
        self.app.post('/login', data={'username': 'test_user', 'password': 'testpass123'})
        # Test computation of Form Accuracy from valid rep frame ratios vs warnings
        res = self.app.post(
            '/save_workout',
            json={
                'exercise_type': 'Push-Ups',
                'rep_count': 12,
                'total_duration_seconds': 40.0,
                'valid_frames': 85,
                'error_warnings': 15
            }
        )
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        # 85 / (85 + 15) = 85.0%
        self.assertEqual(data['session']['avg_form_accuracy'], 85.0)

    def test_login_and_register_pages(self):
        res_login = self.app.get('/login')
        self.assertEqual(res_login.status_code, 200)
        self.assertIn(b'Welcome Back', res_login.data)
        self.assertIn(b'auth-card', res_login.data)

        res_reg = self.app.get('/register')
        self.assertEqual(res_reg.status_code, 200)
        self.assertIn(b'Create Account', res_reg.data)
        self.assertIn(b'auth-card', res_reg.data)


if __name__ == '__main__':
    unittest.main()

