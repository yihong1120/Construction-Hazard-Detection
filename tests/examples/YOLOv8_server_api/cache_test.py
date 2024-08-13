import unittest
from examples.YOLOv8_server_api.cache import user_cache

class CacheTestCase(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment before each test.
        """
        # Clear the cache before each test to ensure a clean slate
        user_cache.clear()

    def test_add_to_cache(self):
        """
        Test adding a user to the cache.
        """
        user_cache['user1'] = 'data1'
        self.assertIn('user1', user_cache)
        self.assertEqual(user_cache['user1'], 'data1')

    def test_remove_from_cache(self):
        """
        Test removing a user from the cache.
        """
        user_cache['user1'] = 'data1'
        del user_cache['user1']
        self.assertNotIn('user1', user_cache)

    def test_update_cache(self):
        """
        Test updating a user in the cache.
        """
        user_cache['user1'] = 'data1'
        user_cache['user1'] = 'data2'
        self.assertEqual(user_cache['user1'], 'data2')

    def test_clear_cache(self):
        """
        Test clearing the entire cache.
        """
        user_cache['user1'] = 'data1'
        user_cache['user2'] = 'data2'
        user_cache.clear()
        self.assertEqual(len(user_cache), 0)

if __name__ == '__main__':
    unittest.main()
