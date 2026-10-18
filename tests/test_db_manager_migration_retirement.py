"""Regression tests for retiring automatic schema migrations."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

from betalens_db_manager.__main__ import main
from betalens_db_manager.manager import DatabaseManager
from betalens_db_manager.schema import SchemaManager, MigrationDisabledError, load_migrations
from betalens_db_manager.import_manifest import ManifestRunner


class MigrationRetirementTests(unittest.TestCase):
    def test_retired_api_never_connects_or_creates_databases(self):
        manager = object.__new__(DatabaseManager)
        with patch('psycopg2.connect', side_effect=AssertionError('must not connect')):
            for operation in (manager.bootstrap, manager.bootstrap_local, manager.verify,
                              manager.verify_schema, manager.plan_migration,
                              SchemaManager.bootstrap_local,
                              SchemaManager.verify_schema, SchemaManager.ensure_schema,
                              load_migrations):
                with self.subTest(operation=operation):
                    with self.assertRaises(MigrationDisabledError):
                        operation()

    def test_retired_cli_fails_before_creating_manager(self):
        with patch('betalens_db_manager.__main__._make_manager', side_effect=AssertionError):
            for args in (['init', '--yes'], ['verify'], ['verify', '--deep']):
                output = io.StringIO()
                with contextlib.redirect_stderr(output):
                    self.assertEqual(main(args), 1)
                self.assertIn('已停用', output.getvalue())

    def test_missing_table_does_not_pass_ensure_dataset(self):
        manager = object.__new__(DatabaseManager)
        manager.schema = Mock()
        manager.schema.inspect_schema.return_value = {'ok': False, 'errors': ['missing daily_market']}
        with self.assertRaisesRegex(RuntimeError, 'missing daily_market'):
            manager.ensure_dataset('daily_market')
        manager.schema.bootstrap_local.assert_not_called()

    def test_inspection_requires_structure_even_when_connection_succeeds(self):
        schema = object.__new__(SchemaManager)
        schema.db_config = {'dbname': 'empty'}
        conn = Mock()
        cur = Mock()
        cur.fetchone.return_value = None
        conn.cursor.return_value.__enter__ = Mock(return_value=cur)
        conn.cursor.return_value.__exit__ = Mock(return_value=False)
        schema.connect = Mock(return_value=conn)
        result = schema.inspect_schema(['trade_calendar'])
        self.assertTrue(result['database_exists'])
        self.assertFalse(result['ok'])
        self.assertTrue(result['errors'])
        conn.set_session.assert_called_once_with(readonly=True, autocommit=True)
        conn.close.assert_called_once()

    def test_connection_failure_cannot_pass_inspection(self):
        schema = object.__new__(SchemaManager)
        schema.db_config = {'dbname': 'missing'}
        schema.connect = Mock(side_effect=RuntimeError('missing database'))
        result = schema.inspect_schema()
        self.assertFalse(result['ok'])
        self.assertFalse(result['database_exists'])
        self.assertEqual(result['errors'], ['missing database'])

    def test_old_manifest_plan_must_be_preflighted_again(self):
        runner = ManifestRunner(job_runner=Mock(), checkpoint_store=Mock(), target_database='datafeed')
        self.assertIsNone(runner.schema_version)
        payload = {'path': '/tmp/retired.yaml', 'token': 'test', 'entries': [],
                   'target_database': 'datafeed', 'schema_version': 10}
        with self.assertRaisesRegex(ValueError, '重新预检'):
            runner.run(payload)
        payload['schema_version'] = None
        self.assertIsNone(runner.from_dict(payload).schema_version)


if __name__ == '__main__':
    unittest.main()
